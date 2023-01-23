from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import numpy as np
import pandas as pd
import torch
import random
import numpy as numpy
import os

# Seeding all
import random
import os
seed = 42
random.seed(seed)
np.random.seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(seed)


#Select model from either of these pre-trained models Coder, Sapbert or Scibert to generate embeddings.
model_used = "Coder"

if model_used == "Sapbert":
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

elif model_used == "Scibert":
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

else:
    model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
    tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')


def get_embeddings(text_list,num_labels):

    embeddings_array=np.zeros((len(text_list),768))
    
    for idx in range (len(text_list)):

        # Tokenize considering the max length of ClinicalBERT - Potential TODO split notes in 512 segments and compute centroid instead
        inputs = tokenizer(text_list[idx], return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        output = model(input_ids, attention_mask=attention_mask)[0]
        embeddings = output[:].mean(dim=1)
        cls=embeddings.detach().numpy()
        cls=np.stack(cls,axis=0)
        cls=np.squeeze(cls)
        embeddings_array[idx]=cls
    
    All_embeddings=embeddings_array
    queries = All_embeddings[:-num_labels]
    labels = All_embeddings[-num_labels:]
    queries = torch.from_numpy(queries)
    labels = torch.from_numpy(labels)
    return queries,labels


def closest_label(sentence_representation, label_representations):
    """Returns the closest label and score for the passed sentence."""
    similarities = F.cosine_similarity(sentence_representation, label_representations)
    closest = similarities.argsort(descending=True)
    return similarities, closest

def get_MRR(rank_df,query_df,dataset_name):
    """Inputs are rank dataframe, query dataframe and the name of the dataset to be evaluated either mimic or dc3."""
    """Returns the MRR of a given dataset, when the inputs described above are provided."""
    true_rank = []
    if dataset_name == 'mimic':
        for i, row in rank_df.iterrows():
            true = row["Truth"]
            a = row[1:]
            idx = a.tolist().index(true) + 1
            true_rank.append(idx)
        MRR_true = np.mean(1 / np.array(true_rank, dtype=float))
        return MRR_true
    
    elif dataset_name == 'dc3':
        query_ids = query_df.Query_ID.tolist()
        lst = list(set(query_ids))
        for i, row in rank_df.iterrows():
            true = row['Truth']
            a = row[1:]
            idx = min([i for i,value in enumerate(a.tolist()) if value == true]) + 1
            true_rank.append(idx)
        rank_list = [min([b for f, b in zip(query_ids, true_rank) if f == i]) for i in lst]
        MRR_true = np.mean(1 / np.array(rank_list, dtype=float))
        return MRR_true


#The queries should have its fullname as ground truth
# queries = pd.read_csv("Dastasets/Mimic_zeroShot_data.csv")
queries = pd.read_csv("Datasets/Filtered_DC3_Data.csv",encoding='latin-1')

queries =queries.dropna()

queries=queries.reset_index(drop=True)

query_ids = queries.Query_ID.tolist()
unique_queries=len(queries.Query_ID.value_counts())

#The full names and true label of the each query 
query_truth = queries.title.tolist()

#read all label full names
labels = queries.title.tolist()

label_dicts={k: v for v, k in enumerate(labels)}

ground_truth = [*map(label_dicts.get, query_truth)]

sentences=queries['text'].tolist()

text_list=sentences+labels
num_categories=len(labels)
sentence_rep,label_rep = get_embeddings(text_list,num_categories)

#create a dataframe by sorting labels according to cosine similarity scores
preds_list = []
for i in range (len(sentence_rep)):
    _, sorted_indices = closest_label(sentence_rep[i],label_rep)
    preds_list.append(sorted_indices.tolist())    
df_Rank = pd.DataFrame(preds_list)

df_Rank["Truth"]=ground_truth
first_column = df_Rank.pop('Truth')
df_Rank.insert(0, 'Truth', first_column)

MRR = get_MRR(df_Rank,queries,"dc3")