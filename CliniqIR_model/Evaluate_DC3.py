import pandas as pd
import numpy as np
import collections
from scipy.stats import rankdata

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
    # print(f"{data.isna().sum().sum()} unique queries")
    data.dropna(inplace=True)
    return data

def UpdateResults(results, gtruth):
    truth_lists = [] 
    label_lists = []
    text_lists = []
    
    Query_label_lists = []
    Query_truth_lists = []
    Query_text_lists = []
    
    truth = gtruth.cui_code.tolist()
    label = gtruth.title.tolist()
    text = gtruth.filtered_text.tolist()

    # strip quotes off the truth list and save to a new list

    for i in range(len(truth)):
        truth_lists.append(truth[i])
        label_lists.append(label[i])
        text_lists.append(text[i])

    # get query ids for each note in a list

    ID = results.QUERYID.tolist()

    # get the true labels for each note query

    for i in range(len(ID)):

        # print(i)

        a = truth_lists[ID[i] - 1]
        e = label_lists[ID[i] - 1]
        b = text_lists[ID[i] - 1]

        Query_truth_lists.append(a)
        Query_label_lists.append(e)
        Query_text_lists.append(b)

    # add the column to the dataframe

    results['Truth'] = pd.Series(Query_truth_lists).values
    results['Title'] = pd.Series(Query_label_lists).values
    results['Text'] = pd.Series(Query_text_lists).values

    return results

def Filter_by_Concepts(dicts,accepted_lists):
    if (isinstance(dicts, str)):
        dicts=eval(dicts)
    else:
        dicts=dicts
    new_dict = dict((k, dicts[k]) for k in accepted_lists if k in dicts)
    return new_dict

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

def Filter_Rows(df,unique_val):
    unique_val=set(unique_val)
    df["DF"]=df['Truth'].apply(lambda x: bool(unique_val.intersection(x)))
    df = df.drop_duplicates(subset='QUERYID', keep="first").reset_index(drop=True)
    df= df.loc[df['DF'] == True]
    df = df.drop_duplicates(subset='Text', keep="first").reset_index(drop=True)
    return df

def Compute_Ranks_by_TFIDF(dataxx,Collection_Size,IDF_dicts):   
    concept_freq=dataxx.P_Frequency_Filtered_Concepts.tolist()
    TFIDF_list2 = []
    TF_IDF_dict2 = {}
    Query_ranks = []
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
    for i in range (len(TFIDF_list2)):
            a=TFIDF_list2[i]
            s_data = sorted(a.items(), key=lambda item: item[1], reverse=True)
            rank, count, previous, result = 0, 0, None, {}
            for key, num in s_data:
                count += 1
                if num != previous:
                    rank += count
                previous = num
                count = 0
                result[key] = rank
            #result2 = dict(zip(result.keys(), rankdata([i for i in result.values()], method='average'))) 
            result2 = dict(zip(result.keys(), rankdata([i for i in result.values()], method='min')))
            Query_ranks.append(result2)
    dataxx['TFIDF_Rank'] = pd.Series(Query_ranks).values
    return dataxx

from statistics import mean
def getMRR(df, Rankcol):
    rank_1 = []
    rank_list=[]
    ranks=[]
    Query_results = df.eval(Rankcol).tolist()
    truths = df.Truth.tolist()
    for i in range (len(Query_results)):
        rank_1=[]
        tru_val = truths[i] #true parent
        for items in tru_val:
            tmp=[value for key, value in Query_results[i].items() if key == items]
                # print(tmp)
            rank_1.append(tmp)
        rank_1=[item for sublist in rank_1 for item in sublist]
        if rank_1 ==[]:
            rank = 0
        else:
            rank =1/(min(rank_1))
        rank_list.append(rank)
    MRR = mean(rank_list)
    print("The MRR by " +  Rankcol + ":", MRR)
    return MRR


df = pd.read_csv("../Datasets/Filtered_DC3_Data.csv",encoding='latin-1')

#read the results file
data = pd.read_csv("CliniqIR_model/Query_Results.txt",sep = "\t", names = ['Abstract_Rank','PMID', 'AbstractTitle',"TF_SCORE","Concepts"])

data =  prepare_query_results(data)

df['cui_code']=df['cui_code'].str.split(',')
accepted_list = [item for sublist in df['cui_code'].tolist() for item in sublist]

# #Filter by unique labels
data['Filtered_Concepts'] = data.apply(lambda x: Filter_by_Concepts(x.Concepts, accepted_lists=accepted_list), axis=1)

data=UpdateResults(data, df)

data = EvaluatebyConcepts(data,"Filtered_Concepts",num_Abstracts=100)

# #select data based on conditions.
pub = pd.read_csv("/CliniqIR/Datasets/DC3_data_pubmed_frequency.csv")
IDF_dicts=dict(zip(pub.concept, pub.Frequency_across_Pubmed))
pub=pub.loc[pub['Frequency_across_Pubmed'] >=100]
unique_val=pub.concept.values.tolist()

data = Filter_Rows(data,unique_val)

Collection_Size = 33000000
data= Compute_Ranks_by_TFIDF(data,Collection_Size,IDF_dicts)

mrr = getMRR(data, "TFIDF_Rank")
