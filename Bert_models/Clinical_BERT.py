'''
This code fine-tunes a pre-trained bert based model on a train and a val dataset.
Then it reads in a list of test data file names, iterates through the list, reads in the data from each file, 
calculates the accuracy, F1-score, and the mean reciprocal rank. The results are printed for each data file.
It also saves the returned rank list for each test query in a dataframe and returns a csv file. 
The rank list file is input required for the RRF Ensemble with CliniqIR.
'''

import pandas as pd
from datasets import Dataset
import numpy as np
import itertools
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, auc
from datasets import DatasetDict
from datasets import load_metric
import torch
import torch.nn.functional as nnf
from transformers import Trainer
from torch import nn
import matplotlib.pyplot as plt
from itertools import cycle
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments

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


# Read the data and shuffle it
val = pd.read_csv("Datasets/Mimic_val.csv")
val = val.sample(frac=1).reset_index(drop=True)
train = pd.read_csv("Datasets/Mimic_train.csv")
train = train.sample(frac=1).reset_index(drop=True)

# Get the target variable
target = train["label"].to_numpy()
class_sample_count = np.unique(target, return_counts=True)[1]
minimum_class = np.amin(class_sample_count)
weight = 1. / class_sample_count

# Calculate custom weight for each class
weight = weight * minimum_class
class_weight = torch.tensor(weight, dtype=torch.float, device='cuda:0')

# Prepare the dataset
Train_data = Dataset.from_pandas(train)
Val_data = Dataset.from_pandas(val)

def tokenize_function(examples):
    return tokenizer(examples['text'], max_length=512, padding='max_length', truncation=True)

full_dataset = DatasetDict({'train': Train_data, 'val': Val_data})

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")
tokenzied_dataset = full_dataset.map(tokenize_function, batched=True)

full_train_dataset = tokenzied_dataset["train"].shuffle(seed=42)
full_eval_dataset = tokenzied_dataset["val"].shuffle(seed=42)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT", num_labels=len(class_sample_count))

model.resize_token_embeddings(len(tokenizer))

# Set the training arguments
training_args = TrainingArguments("ClinincalBERT_trainer", num_train_epochs =20, learning_rate = 1e-05,evaluation_strategy= "steps",load_best_model_at_end= True)

# Define the metrics function
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1_macro = f1_score(y_true=labels, y_pred=pred, average="macro")
    return {"accuracy": accuracy, "f1_macro":f1_macro}

# Define the custom trainer and define custom loss to handle class imbalance.
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=class_weight) 
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# Define the function to store ranks and calculate MRR
"""
This function takes in two inputs, probabilities and true_labels and returns the mean reciprocal rank by sorting the probabiliites.  
It then calculates the rank of the true labels in the sorted array and finds the mean reciprocal rank. 
It saves the sorted rank array to a csv file.
The function also prints the mean reciprocal rank.
"""

def getRank(probs, true_labels):
    sorted_probs = torch.sort(probs, 1, descending=True)
    sorted_arr = sorted_probs.indices.cpu().detach().numpy()
    probs_arr = probs.cpu().detach().numpy()

    # Saving the sorted array to a csv file
    dfX = pd.DataFrame(sorted_arr)
    dfX.to_csv("/Datasets/Clinical_BERT_ranks.csv", index=False)

    rank_list = []
    for i in range(len(true_labels)):
        rank = ((sorted_probs.indices[i] == true_labels[i]).nonzero(as_tuple=True)[0])
        y = rank.tolist()
        rank_list.append(y)

    flat_list = list(itertools.chain(*rank_list))
    new_rank_list = [x+1 for x in flat_list]
    df = pd.DataFrame(new_rank_list, columns =['Rank'])
    df['Reciprocal_Rank'] = df.apply(lambda x: 1/x['Rank'] if x['Rank']>0 else x['Rank'], axis=1)
    MRR = df[["Reciprocal_Rank"]].mean()
    #print("The Mean Reciprocal Rank by sorting_probs is: ", MRR)
    return MRR

#change to custom trainer
trainer = CustomTrainer(
    model=model, args=training_args, train_dataset=full_train_dataset, eval_dataset=full_eval_dataset)

trainer.train()
metric = load_metric("accuracy")
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


'''
This code reads in a list of data file names, iterates through the list, 
reads in the data from each file, creates a dataset, tokenizes the dataset, 
makes predictions, preprocesses the raw predictions, calculates the accuracy and F1-score, 
and calculates the mean reciprocal rank. The results are printed for each data file.

'''

# Creating a list of data file names
list_data = ["Datasets/Mimic_test.csv"]

# Iterating through the list of data files
for data_file in list_data:
    # Reading the data from the csv file
    test = pd.read_csv(data_file)

    # Creating a dataset from the test data
    test_data = Dataset.from_pandas(test)
    test_dataset = DatasetDict({'test': test_data, 'test': test_data})

    # Tokenizing the dataset
    tokenized_data = test_dataset.map(tokenize_function, batched=True)
    full_test_dataset = tokenized_data["test"]

    # Predictions and true labels
    raw_pred1, true_labels1, _ = trainer.predict(full_test_dataset)

    # Preprocessing raw predictions
    y_pred1 = np.argmax(raw_pred1, axis=1)
    raw_pred1 = torch.tensor(raw_pred1)
    probs = nnf.softmax(raw_pred1, dim=1)

    # Accuracy and F1-score
    accuracy = accuracy_score(true_labels1, y_pred1)
    f1_macro = f1_score(true_labels1, y_pred1, average="macro")

    # Mean Reciprocal Rank
    rank1 = getRank(probs, true_labels1)

    # Printing the results
    print(f"Acc for {data_file} is: {accuracy} f1_macro is: {f1_macro}")
    print(f"MRR for {data_file} is: Mean {rank1}")
