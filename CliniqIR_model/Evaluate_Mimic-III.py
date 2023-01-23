import pandas as pd
import numpy as np
import json
import collections
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import json
import collections
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score

def prepare_query_results(data):
    # Fill empty abstract titles with 'unknown'
    data['AbstractTitle'] = data['AbstractTitle'].fillna(value='unknown')
    data.dropna(inplace=True)

    # Assign unique ids to the query results based on their query id
    count = 0
    unique_query_ids = []
    for i in data['Abstract_Rank']:
        if i == "QueryID":
            unique_query_ids.append(None)
            count += 1
        else:
            unique_query_ids.append(count)

    # Append unique query ids to dataframe
    data['QUERYID'] = unique_query_ids
    # Convert query column to numeric data type
    data['QUERYID'] = data['QUERYID'].astype('Int64')
    print(f"{data.isna().sum().sum()} unique queries")
    data.dropna(inplace=True)
    return data


import json

# Function to filter concept dictionary by semtypes or filter by concepts filter.
#you have to provide it a list of accepted types.
def Filter_by_Concepts(dicts,accepted_lists):
    if (isinstance(dicts, str)):
        dicts=eval(dicts)
    else:
        dicts=dicts
    new_dict = dict((k, dicts[k]) for k in accepted_lists if k in dicts)
    return new_dict

def Filter_by_Sem_Types(dicts,accepted_lists):
    if (isinstance(dicts, str)):
        dicts=eval(dicts)
    else:
        dicts=dicts
    filtered_dict = {k:v for (k,v) in dicts.items() if any (s in v for s in accepted_lists)}
    return filtered_dict

def getNConcepts(col):
    col=eval(col)
    dicts={A:N for (A,N) in [x for x in col.items()][:1]}
    return dicts

def getLabels(df,category):
    labels = df.eval(category).tolist()
    labels = list(set(labels))
    # print("This category contains ", len(labels), "unique labels")
    return labels

def UpdateResults(results, gtruth):
    truth_lists = [] 
    label_lists = []
    freq_lists = []
    
    Query_label_lists = []
    Query_truth_lists = []
    Query_freq_lists = []
    
    truth = gtruth.cui_code.tolist()
    label = gtruth.label.tolist()
    freq = gtruth.freq.tolist()

    # strip quotes off the truth list and save to a new list

    for i in range(len(truth)):
        truth_lists.append(truth[i])
        label_lists.append(label[i])
        freq_lists.append(freq[i])

    # get query ids for each note in a list

    ID = results.QUERYID.tolist()

    # get the true labels for each note query

    for i in range(len(ID)):

        # print(i)

        a = truth_lists[ID[i] - 1]
        e = label_lists[ID[i] - 1]
        b = freq_lists[ID[i] - 1]

        Query_truth_lists.append(a)
        Query_label_lists.append(e)
        Query_freq_lists.append(b)

    # add the column to the dataframe

    results['Truth'] = pd.Series(Query_truth_lists).values
    results['label'] = pd.Series(Query_label_lists).values
    results['Freq'] = pd.Series(Query_freq_lists).values

    return results

# select rows whose ground truth meet a certain criteria;
#give in the list of groundtruth that meets the condition
def Filter_Rows(dataxx, unique_val):
    dataxx=dataxx[dataxx['Truth'].isin(unique_val)]
    dataxx=dataxx.reset_index(drop=True)
    #reset queryids
    temp = dataxx.QUERYID.tolist()
    count = 1
    new_list = []
    for index, elem in enumerate(temp):
        if (index+1 < len(temp) and index - 1 >= 0):
            prev_el = (temp[index-1])
            curr_el = (elem)
            if prev_el != curr_el:
                count+=1
        new_list.append(count)
    dataxx['QUERYID'] = pd.Series(new_list).values
    return dataxx


#computes accuracy for ground truth returned by the model
def get_correct_predictions(df,category):
    boolean_list=[]
    final=[]
    truth= df.Truth.tolist()
    Query_prediction = df.eval(category).tolist()
    
    
    for i in range (len(Query_prediction)):
        z=[a for a in Query_prediction[i] if a in truth[i]]
        final.append(z)
        
        if (len(z)) > 0:
            boolean_list.append(True)
            
        else:
            boolean_list.append(False)
            
    df['correct_prediction'] = pd.Series(final).values
    #calculate recall accuracy by measuring counting if any abstract returns a concept that is equivalent to the ground truth 
    pred_count = sum(boolean_list)
    recall_acc = pred_count / float(len(truth)) * 100.0
    return df


#get IDF of each concept based on number of abstracts collection in each query. 
#here TF is frequency of each concept across absracts and Idf is calculated based on number of abstracts per query
#IDF_dicts contains frequency of each concepts across all pubmed collections
#total pubmed collections is 33million

def Compute_Ranks_by_TFIDF(dataxx,Collection_Size,IDF_dicts):   
    concept_freq=dataxx.P_Frequency_Filtered_Concepts.tolist()
    TFIDF_list2 = []
    TF_IDF_dict2 = {}
    for i in range (len(concept_freq)):
        Dx = list(concept_freq[i].keys())
        TF_IDF_dict2 = {} 
        for key in Dx:
            TF = concept_freq[i][key]
            #print(TF)
            n = IDF_dicts[key]
            IDF = np.log10(Collection_Size/n)
            Tfidf = TF * IDF
            TF_IDF_dict2[key]= Tfidf
        TFIDF_list2.append(TF_IDF_dict2)
    dataxx['TFIDF_Rank'] = pd.Series(TFIDF_list2).values
    return dataxx    


#Evaluate by concepts or filtered concepts
def EvaluatebyConcepts(df,Concepts,num_Abstracts):
    preds = []
    zlist=[]
    concept_lists = df.eval(Concepts).tolist()
    ID = df.QUERYID.tolist()
    truth= df.Truth.tolist()
    
    #create a list of tuples by mapping concepts to unique query ids
    list_of_tuples = list(zip(ID, concept_lists))
   

    #create a list of lists to store predictions of each notes query in each list
    #preds_lists = [ [] for _ in range(len(truth))]
    preds_lists = [ [] for _ in range(int(len(truth)/num_Abstracts))]
    
       
    #store concepts for each query in a list based on its index
    for index, tuple in enumerate(list_of_tuples):
        idx = tuple[0]
        val = tuple[1]
        preds_lists[idx-1].append(val)
   
    #count and aggregate the concepts for each ID
    for _ in range (len(preds_lists)):
       # _ = eval(_)
        if (isinstance(preds_lists[0][0], str)):
        #if preds_lists[_][0]
            datum = [j for i in preds_lists[_] for j in (eval(i)).keys()]
        else:
            datum = [j for i in preds_lists[_] for j in (i).keys()]
        counter=collections.Counter(datum)
        counter=sorted(counter.items(), key=lambda x: x[1], reverse=True)
        adict = dict(counter)
        preds.append(adict)
        temp=list(adict.keys())
        zlist.append(temp)

    #return highest ranked concept predicted for every query with same id
    Query_preds_lists=[]
    Query_preds_dicts=[]
    Query_ranks = []
    Query_ranks2 = []

    for i in range (len(ID)):
        #print(i)
        a =preds[ID[i]-1]
        b =zlist[ID[i]-1]
        Query_preds_lists.append(a)
        Query_preds_dicts.append(b)
        
    df['Predictions_'+Concepts] = pd.Series(Query_preds_dicts).values
    df['P_Frequency_'+Concepts] = pd.Series(Query_preds_lists).values
    df = df.drop_duplicates(subset='QUERYID', keep="first").reset_index(drop=True)
    df=df.reset_index(drop=True)
    
    return df

#Gets ranks of each concept after sorting by frequency of occurence
def getRank(df,Concepts,freq_col):
    Query_ranks=[]
    Query_ranks2=[]
    freq_list=df.eval(freq_col).tolist()
    for i in range (len(freq_list)):
        a=freq_list[i]
        s_data = sorted(a.items(), key=lambda item: item[1], reverse=True)
        rank, count, previous, result = 0, 0, None, {}
        for key, num in s_data:
            count += 1
            if num != previous:
                rank += count
            previous = num
            count = 0
            result[key] = rank
        result2 = dict(zip(result.keys(), rankdata([i for i in result.values()], method='average'))) 
        Query_ranks.append(result)
        Query_ranks2.append(result2)
    df['Rank_byFreq_'+Concepts] = pd.Series(Query_ranks).values #ranks after sorting by frequency of occurrence
    df['Rank_Dicts2_'+Concepts] = pd.Series(Query_ranks2).values
    return df

def GetMRR(df,RankCol1):
    resultsX=df.correct_prediction.tolist()
    Query_ranks=[]
    Query_ranks2=[]
    answers =[]
    Query_list=df.eval(RankCol1).tolist()
    for i in range (len(Query_list)):
        a=Query_list[i]
        s_data = sorted(a.items(), key=lambda item: item[1], reverse=True)
        rank, count, previous, result = 0, 0, None, {}
        for key, num in s_data:
            count += 1
            if num != previous:
                rank += count
            previous = num
            count = 0
            result[key] = rank
        # result2 = dict(zip(result.keys(), rankdata([i for i in result.values()], method='average')))
        result2 = dict(zip(result.keys(), rankdata([i for i in result.values()], method='min')))
        Query_ranks.append(result)
        Query_ranks2.append(result2)
        answers.append(list(result2.keys())[0])
    count = 0 
    temp = 0
    listz = []
    for i in range (len(resultsX)):
        if resultsX[i] == []:
            rnk = 0
            r =0
            listz.append(r)
            
        else:
            count+=1
            ky = resultsX[i][0]
            rank = Query_ranks[i][ky]
            r = rank
            rrank = 1/rank
            temp = temp + rrank
            listz.append(r)
    MRR = temp/(len(resultsX))
    
    print("The CliniqIR MRR by " +  RankCol1 + ":", MRR)
    
    df['Rank_Dicts2_'+RankCol1] = pd.Series(Query_ranks2).values
    df["CliniqIR_predictions_rank"] = listz
    df["CliniqIR_prediction"]=answers
    top1 = len([num for num in listz if num == 1])
    Acc = top1/len(listz) * 100
    print("The Accuracy is for CliniqIR: " , Acc)
    df = df[["QUERYID","label","Truth","Freq","Rank_Dicts2_TFIDF_Rank","CliniqIR_predictions_rank","CliniqIR_prediction"]]
    return MRR, df

def GetBertRank(df, bert_df,label_dicts):
    bert_lists = []
    # map dictionary values to entire dataframe:
    for i in range (len(df.Truth.value_counts())):
        bert_df.iloc[:,i] = bert_df.iloc[:,i].map(label_dicts)

    for i, row in bert_df.iterrows():
        z = row[0:len(df.Truth.value_counts())]
        z = z.values.tolist()
        newz = {k: v+1 for v, k in enumerate(z)}
        bert_lists.append(newz)
    return bert_lists   

def RRF_Ensemble(bert_ranks,cliniqIR_ranks,df):
    RRF_fuse=[]
    result2 = {}
    new_ranks =[]
    answers = []
    concept_dicts = {}
    
    dicts_freq = dict(zip(df.Truth, df.Freq))
    true_concepts = list(df.Truth.values)
    All_concepts =  list(set(df.Truth.values))
    
    for j in range(len(true_concepts)):
        for i in All_concepts: 
            ranka = bert_ranks[j].get(i,100000)
            # rankb = newbertB[j].get(i,100000)
            rankc = cliniqIR_ranks[j].get(i,100000)
            freq_concept = dicts_freq[i]
            if freq_concept <= 1:
                fusion_score = (1/(60 + rankc))
            else:
                fusion_score = (1/(60 + ranka)) + (1/(60 + rankc))#
            concept_dicts[i]=fusion_score
        s_data = sorted(concept_dicts.items(), key=lambda item: item[1],reverse=True)
        rank, count, previous, result = 0, 0, None, {}
        for key, num in s_data:
            count += 1
            if num != previous:
                rank += count
            previous = num
            count = 0
            result[key] = rank
        result2 = dict(zip(result.keys(), rankdata([i for i in result.values()], method='min')))
        answers.append(list(result2.keys())[0])
    
        #get the rank of the correct concept
        truth=true_concepts[j]
        RRF_fuse.append(result2[truth])
    reciprocal_ranks2 = [1/(item) for item in RRF_fuse]
    MRR_RRF =sum(reciprocal_ranks2)/len(reciprocal_ranks2)
        #map cui predictions to label
    Label_dicts=dict(zip(df.Truth, df.label))
    df["ensemble_preds"]=answers
    df["ensemble_preds_label"] = df["ensemble_preds"].map(Label_dicts)
    y_test = df.label.tolist()
    y_preds = df.ensemble_preds_label.tolist()
        # print(metrics.classification_report(y_test, y_preds))
    acc = accuracy_score(y_test, y_preds)
    print("accuracy score is: ", acc)
    print("MRR score is: " ,MRR_RRF)
    return MRR_RRF


test = pd.read_csv("Datasets/Mimic_test.csv")

# map labels to cui_code
Label_dicts=dict(zip(test.label, test.cui_code))

#read in the CliniqIR model results file
data = pd.read_csv('CliniqIR_model/Query_Results.txt', sep = "\t", names = ['Abstract_Rank','PMID', 'AbstractTitle',"TF_SCORE","Concepts"])

#Unique labels in the notes.
unique_labels=getLabels(test,"cui_code")

data =prepare_query_results(data)

#Filter by unique labels
data['Filtered_Concepts'] = data.apply(lambda x: Filter_by_Concepts(x.Concepts, accepted_lists=unique_labels), axis=1)

data = UpdateResults(data, test)

#get Pubmed frequency information of all true_labels and store in a dictionary
pub = pd.read_csv("Datasets/Concept_Frequency_across_Pubmed_Abstract.csv")
IDF_dicts=dict(zip(pub.code, pub.Frequency_across_Pubmed))

# pub=pub.loc[pub['Frequency_across_Pubmed'] >=100]
# #get unique labels that meet this condition.
# unique_val=pub.code.tolist()
# dataxx = Filter_Rows(data, unique_val)

#Evaluate results by using filtered concepts or raw concepts, num_Abstracts = abstract returned per query.
data = EvaluatebyConcepts(data,"Filtered_Concepts",num_Abstracts=100)

#get rank information of filtered conceepts based on their frequency
data=getRank(data,"Filtered_Concepts","P_Frequency_Filtered_Concepts")

#get the correct predictions for evaluation purposes.
data=get_correct_predictions(data,"Predictions_Filtered_Concepts")

#compute ranks by TF_IDF and feed in the concept pubmed mentions dictionary
documents=33000000
data=Compute_Ranks_by_TFIDF(data,documents,IDF_dicts)

#Compute MRR based on TF_IDF ranks
MRR,output_df=GetMRR(data,"TFIDF_Rank")

# df.to_csv("")

#get Clinical BERT's ranks for the same test data
bert_df = pd.read_csv("/Datasets/Clinical_BERT_ranks.csv")

Bert_ranks= GetBertRank(output_df,bert_df,Label_dicts)

CliniqIR_ranks = output_df.Rank_Dicts2_TFIDF_Rank.tolist()

RRF_score=RRF_Ensemble(Bert_ranks,CliniqIR_ranks,output_df)