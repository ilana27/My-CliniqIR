import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Select model from either of these pre-trained models Coder, Sapbert or Scibert to generate embeddings.
model_used = "ClinicalBert"

if model_used == "Sapbert":
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

elif model_used == "Scibert":
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

elif model_used == "ClinicalBert":
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

elif model_used == "PubmedBert":
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

elif model_used == "PubmedBert(abstracts)":
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

else:
    model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
    tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')


def get_embeddings(text_list, num_labels):
    embeddings_array = np.zeros((len(text_list), 768))

    for idx in range(len(text_list)):
        inputs = tokenizer(text_list[idx], return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        output = model(input_ids, attention_mask=attention_mask)[0]
        embeddings = output[:].mean(dim=1)
        cls = embeddings.detach().numpy()
        cls = np.stack(cls, axis=0)
        cls = np.squeeze(cls)
        embeddings_array[idx] = cls

    All_embeddings = embeddings_array
    queries = All_embeddings[:-num_labels]
    labels = All_embeddings[-num_labels:]
    queries = torch.from_numpy(queries)
    labels = torch.from_numpy(labels)
    return queries, labels


def closest_label(sentence_representation, label_representations):
    similarities = F.cosine_similarity(sentence_representation, label_representations)
    closest = similarities.argsort(descending=True)
    return similarities, closest


def get_MRR(rank_df,query_df,dataset_name):
    true_rank = []
    if dataset_name == 'mimic':
        for i, row in rank_df.iterrows():
            true = row["Truth"]
            a = row[1:]
            idx = a.tolist().index(true) + 1
            true_rank.append(idx)
        MRR_true = np.mean(1 / np.array(true_rank, dtype=float))
        print(MRR_true)
        return MRR_true
    elif dataset_name == 'dc3':
        query_ids = query_df.Query_ID.tolist()
        lst = list(set(query_ids))
        for i, row in rank_df.iterrows():
            true = row['Truth']
            a = row[1:]
            idx = min([i for i, value in enumerate(a.tolist()) if value == true]) + 1
            true_rank.append(idx)
        rank_list = [min([b for f, b in zip(query_ids, true_rank) if f == i]) for i in lst]
        MRR_true = np.mean(1 / np.array(rank_list, dtype=float))
        print(MRR_true)
        return MRR_true


#Query pre-processing for DC3
# The queries should have their full name as ground truth

queries = pd.read_csv("Datasets/Filtered_DC3_Data.csv",encoding='latin-1")
queries = queries.dropna()
queries = queries.reset_index(drop=True)
query_ids = queries.Query_ID.tolist()
unique_queries = len(queries.Query_ID.value_counts())
query_truth = queries.title.tolist()
labels = queries.title.tolist()
label_dicts = {k: v for v, k in enumerate(labels)}
ground_truth = [*map(label_dicts.get, query_truth)]
sentences = queries['text'].tolist()
text_list = sentences + labels
num_categories = len(labels)
sentence_rep, label_rep = get_embeddings(text_list, num_categories)
preds_list = []
for i in range(len(sentence_rep)):
    _, sorted_indices = closest_label(sentence_rep[i], label_rep)
    preds_list.append(sorted_indices.tolist())
df_Rank = pd.DataFrame(preds_list)
df_Rank["Truth"] = ground_truth
first_column = df_Rank.pop('Truth')
df_Rank.insert(0, 'Truth', first_column)
MRR = get_MRR(df_Rank, queries,"dc3")

#get all mimic data categories to be evaluated 
file_list = glob.glob("Datasets/Mimic_zeroShot_data/*.csv")
file_list.sort()
file_list = file_list[2:-2]
labels = pd.read_csv("/Datasets/Pubmed100_Class_Names.csv")
labels = labels['classes'].tolist()
label_dicts = {k: v for v, k in enumerate(labels)}
MRR_list = []
for file in file_list:
    preds_list = []
    queries = pd.read_csv(file)
    queries = queries.dropna()
    queries = queries.reset_index(drop=True)
    query_truth = queries.title.tolist()
    ground_truth = [*map(label_dicts.get, query_truth)]
    sentences = queries['text'].tolist()
    text_list = sentences + labels
    num_categories = len(labels)
    x, y = get_embeddings(text_list, num_categories)
    print(type(x))
    for i in range(len(x)):
        _, sorted_indices = closest_label(x[i], y)
        preds_list.append(sorted_indices.tolist())
    df_Rank = pd.DataFrame(preds_list)
    df_Rank["Truth"] = ground_truth
    first_column = df_Rank.pop('Truth')
    df_Rank.insert(0, 'Truth', first_column)
    MRR = get_MRR(df_Rank,queries, "mimic")
    MRR_list.append(MRR)
    print(file, MRR)
