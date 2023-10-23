import configparser
import os
from sshtunnel import SSHTunnelForwarder
import psycopg2
import pandas as pd
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split


'''
This code collect Mimic-III Discharge Summaries from a database given your login credentials.
'''

# Read configuration file and establish SSH tunnel
config = configparser.ConfigParser()
config_file = os.path.join(os.getcwd(),'config', 'credentials.ini')
config.read(config_file)

with SSHTunnelForwarder((config['ssh']['ip'], int(config['ssh']['port'])),
            ssh_username=config['ssh']['username'],
            ssh_password=config['ssh']['password'],
            remote_bind_address=('localhost', int(config['postgres']['port'])),
            local_bind_address=('localhost', int(config['postgres']['port']))):
                print ("I got in")

# Connect to PostgreSQL
con = psycopg2.connect(dbname=config['postgres']['dbname'], user=config['postgres']['user'], password=config['postgres']['password'], host=config['postgres']['host'], port=config['postgres']['port'])

# Define SQL queries
admissions_query = "(select hadm_id, diagnosis from mimiciii.admissions);"
notes_query = "(select * from mimiciii.noteevents WHERE category = 'Discharge summary');"
diagnosis_query = "(select subject_id, hadm_id, icd9_code from mimiciii.diagnoses_icd WHERE seq_num ='1');"
icd_codes_query = "(select subject_id, hadm_id, icd9_code from mimiciii.diagnoses_icd)"
dicd_query = "(select * from mimiciii.d_icd_diagnoses);"

# Execute SQL queries and store results in DataFrames
retrieved_notes = pd.read_sql_query(notes_query, con)
icd_codes = pd.read_sql_query(icd_codes_query, con)
admissions = pd.read_sql_query(admissions_query, con)
dcid= pd.read_sql_query(dicd_query, con)
diagnosies = pd.read_sql_query(diagnosis_query, con)

# Drop duplicates and merge data
diagnosies = diagnosies.drop_duplicates(subset=['hadm_id'])
heartset = pd.merge(diagnosies, admissions, how='left', on='hadm_id', validate='one_to_one')

# Remove duplicate notes and merge data
retrieved_notes2 = retrieved_notes.copy()
retrieved_notes2 = retrieved_notes2.drop_duplicates(subset=['text'])
retrieved_notes2 = retrieved_notes2.sort_values(by=['hadm_id'])
retrieved_notes2 = retrieved_notes2.reset_index(drop=True)

retrieved_notes3 = retrieved_notes2.copy()
size = retrieved_notes3.shape[0]
idx = 0

while idx < size-1 :
    if retrieved_notes3['hadm_id'][idx+1]==retrieved_notes3['hadm_id'][idx]:
        retrieved_notes3.at[idx+1,'text']=retrieved_notes3['text'][idx]+retrieved_notes3['text'][idx+1]
        retrieved_notes3=retrieved_notes3.drop([idx])
    idx=idx+1    

heartset = pd.merge(diagnosies,admissions,how='left',on='hadm_id',validate='one_to_one')
notes = pd.merge(heartset,retrieved_notes3,how='inner',on='hadm_id')
notes1 = pd.merge(notes,dcid,how='left',on='icd9_code')

#drop irrelevant labels. 
cols = [0,4,5,6,7,8,9,10,11,12,14]
notes1.drop(notes1.columns[cols],axis=1,inplace=True)


"""
Pre-process the retrieved notes, add periods to ICD code, Maps ICD codes to CUI codes given a CUItoICD file.
"""

#adds period to ICD codes from MIMICIII
def ICDPeriodInsert(x):
    if len(x)==3:
        return x
    else:
        return (x[0:3]+"."+ x[3:])   

notes1['icd9_code1'] = notes1['icd9_code'].apply(ICDPeriodInsert)

df2 = pd.read_csv("~/Datasets/ICD_CUI_files.csv")
df2 = df2.rename({'icd_code1': 'icd9_code'}, axis=1)
notes1 = notes1.rename({'icd9_code': 'icd9_code0', 'icd9_code1': 'icd9_code' }, axis=1)

#merge notes1 and df2 on 'icd9_code' column
Query = notes1.merge(df2, on='icd9_code', how='left')

#drop duplicate rows in the 'text' column
Query = Query.drop_duplicates(['text'],keep= 'first')

#filter rows in the 'title' column based on certain keywords
list1=['birth', 'pregnancy', 'jaundice','fetal','pregnancies', 'abortion','stillborn','liveborn', 
'cesarean section', 'preterm', 'infant', 'childbirth', 'newborn', '33-34 completed weeks of gestation',
'Extreme immaturity, 500-749 grams', 'Fetal growth retardation, unspecified, 2,500', 
"Neonatal hypoglycemia", 'neonatal','Other infection specific to the perinatal period','Neonatal bradycardia']

unique_cols = (Query['title'].unique()).tolist()
filter_data = [x for x in unique_cols if all(y not in x for y in list1)]
Query = Query[Query['title'].isin(filter_data)]

#reset index of the filtered dataframe
Query=Query.reset_index(drop=True)


"""
Maps ICD codes and CUI codes to their semantic types given the MRSTY.RRF file.
"""

#define list of matches
matches = ['T047', 'T037','T061','T033','T191','T046','T060','T048','T019','T020','T184','T068','T190','T031','T101']

#define function to filter lines in a file
def Semantic_Search(line):
    if any(x in line for x in matches):
        return line

#open and read a file, filter the lines based on the function and store the filtered lines in a list
with open("MRSTY.RRF") as f:
    lines = [line.rstrip('\n') for line in f]
an_iterator = filter(Semantic_Search, lines)
filtered_list = list(an_iterator)

#convert the 'cui_code' column to a list
cuiList = Query.cui_code.tolist()
semList = [ [] for _ in range(len(cuiList))]
semList2=[]

#iterate through the filtered lines and extract the cui and sem, and store them in a list
for i in range (len(filtered_list)):
    cui=filtered_list[i].split("|")[0]
    sem=filtered_list[i].split("|")[1]
    a = list(filter(lambda x: cui in x, cuiList))
    if len(a) > 0:
        semList2.append((cui,sem))

data = pd.DataFrame(semList2, columns=['cui_code', 'semtype'])
Query2=Query.merge(data, on='cui_code', how='left')
Query2 = Query2.drop_duplicates(['text'],keep= 'first')

"""
Function to remove uninformative characters from text
"""

def remove_uninformative_characters(x):
    y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
    y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('date of discharge:','',y)
    y=re.sub('--|__|==','',y)
    #y=re.sub('*','a',x)
    y=re.sub('%','',y)
    y=re.sub('>','',y)
    y=re.sub('<','',y)
    y=re.sub('#','',y)
    y=re.sub('_','',y)
    #y=re.sub(')','',y)
    #y=re.sub('(','',y)
    #y=re.sub('+','',y)
    
    
    # remove, digits, spaces
    y = y.translate(str.maketrans("", "", string.digits))
    y = " ".join(y.split())

    # MIMIC-specific preprocessing
    y=re.sub('date of birth:','',y)
    y=re.sub('sex: m','',y)
    y=re.sub('sex: f','',y)
    
    # Some more preprocessing I'm not entirely sure about
    y=re.sub('incomplete dictation','',y)
    y=re.sub('dictator hung up','',y)
    y=re.sub('dictated by: medquist','',y)
    y=re.sub('completed by:','',y)
    y=re.sub('date medicine allergies','',y)
    y=re.sub('medicine allergies','',y)
    y =re.sub(' +', ' ', y)
    y = re.sub('\W+',' ', y)
    
    return y

def remove_voids_and_uninformative_characters(notes,category): 
    notes[category]=notes[category].fillna(' ')
    notes[category]=notes[category].str.replace('\n',' ')
    notes[category]=notes[category].str.replace('\r',' ')
    notes[category]=notes[category].apply(str.strip)
    notes[category]=notes[category].str.lower()

    notes[category]=notes[category].apply(lambda x: remove_uninformative_characters(x))
    
    return notes
"""
Prepare labels based on frequency of unique diagnoses
"""
def add_frequency_position(df, column_name):
    # Count the frequency of each unique value in the column
    freq = df[column_name].value_counts()
    # Create a new dataframe with the unique values and their frequency
    freq_df = pd.DataFrame({column_name: freq.index, 'frequency': freq.values})
    # add new column with the position of each unique column according to it's frequency size
    freq_df['label'] = freq_df['frequency'].rank(method='first', ascending=False) - 1
    freq_df['label'] = freq_df['label'].astype('int')
    # Merge the new dataframe with the original dataframe on the column of interest
    df = pd.merge(df, freq_df, on=column_name)
    return df

category = 'text'
Query2=remove_voids_and_uninformative_characters(Query2,category)
Query2 = add_frequency_position(Query2, 'cui_code')

#select queries with only one frequency for the test set.
Ones = Query2.loc[Query2['frequency'] == 1]
new_Queries = Query2.drop(Query2.index[Query2['frequency'] ==1])

def split_imbalanced_df(df):
    # Get unique labels
    labels = np.unique(df["label"])
    
    # Initialize empty lists to store split data
    train_data = []
    test_data = []
    
    # Iterate over unique labels
    for label in labels:
        # Get all rows with current label
        label_rows = df[df["label"] == label]
        
        # Split label data into train and test sets
        train, test = train_test_split(label_rows, test_size=0.3, random_state=42, stratify=label_rows["label"])
        
        # Append train and test data to respective lists
        train_data.append(train)
        test_data.append(test)
    
    # Concatenate all split data into single dataframes
    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    
    #make sure we all unique labels in the test set
    test_df['Counts'] = test_df['label'].map(test_df['label'].value_counts())
    Twos= test_df.loc[(test_df['Counts'] <= 2) | (test_df['frequency'] <= 3)]
    N_test_df = test_df.drop(test_df.index[(test_df['Counts'] <=2) | (test_df['frequency'] <= 3)])
    
    # Split test data into test and validation sets
    testX_df, val_df = train_test_split(N_test_df, test_size=0.5, random_state=42, stratify=N_test_df['label'])
    
    # Concatenate all stored test data into a single dataframe
    Final_test = pd.concat([testX_df, Twos])
    
    return train_df, Final_test, val_df

train_df, test_df, val_df = split_imbalanced_df(new_Queries)
Final_test = pd.concat([test_df, Ones])

#Prepare input data for Clinical BERT
train_df[["text","label"]].to_csv("Mimic_train.csv",index=False)
# Final_test[["text","label"]].to_csv("Mimic_test.csv",index=False)
val_df[["text","label"]].to_csv("Mimic_val.csv",index=False)
Final_test.to_csv("Mimic_test.csv")

#Prepare input data for Zero-shot baselines
Final_test[["text","title"]].to_csv("test_data_zero_shot.csv",index=False)

#Prepare input data for CliniqIR
QueryTest = Final_test['text'].apply(lambda x: ' '.join(x.split()[1:701])) # 701 words, if only working with ten DC3 notes, okay to use whole thing
np.savetxt('CliniqIR_model/Queries.txt', QueryTest.values, fmt = "%s") #save it in a specific format to Queries.txt
#Keep all columns for evaluating CliniqIR
Final_test.to_csv("Mimic_test.csv")
