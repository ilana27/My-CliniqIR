import glob
import pandas as pd
import time
from collections import Counter


"""
Given the dictionary of concepts extracted for each PubMed abstracts;
this code gets the frequency of PubMed mentions of each concept specified in a list. 
"""

#fxn to get the unique labels (cui_codes)from test set..
def getLabels(df,category):
    #category = eval(category)
    labels = df.eval(category).tolist()
    labels = list(set(labels))
    print("This category contains ", len(labels), "unique labels")
    return labels

#To get unique labels in the notes.

# notes = pd.read_csv("Mimic_Test.csv")
# unique_labels=getLabels(notes,"cui_code")

unique_labels =["C0600260","C0151332"]

def getDictionary(file):
    myDict = {}
    start_time = time.time()
    filename=(file.rsplit("/",1)[1]).rsplit('.',1)[0]+"_Dict.txt"
    outputdir= "Datasets/Dictionary/"
    for ky in unique_labels:
        with open(file) as infile:
            temp = 0
            for line in infile:
                X=eval(line)
                if ky in X:
                    temp = temp+1
            myDict[ky]=temp
    #store dicts
    with open(outputdir+filename,'w') as data: 
        data.write(str(myDict))
    elapsed_time = (time.time() - start_time) 
    #print(elapsed_time)
    print("Done with concept extraction")

file_list=[]
for files in glob.iglob('CliniqIR_model/Pubmed_Concepts/*.txt'):
    file_list.append(files)

#call function with out multi-threaded processes.
for i in file_list:
    getDictionary(i)

# from multiprocessing import Pool
# if __name__ == '__main__':
#     with Pool(20) as p:
#         p.map(getDictionary, file_list)
#         p.close() # no more tasks
#         p.join()
# print("Done with Everything")

# Count the PubMed mentions and store the frequency of each concept in a dataframe 
file_list2=[]
dict_lists=[]
for files in glob.iglob('Datasets/Dictionary/*Dict.txt'):
    file_list2.append(files)

for file in file_list2:
    with open(file) as infile2: 
        for line in infile2:
            d = eval(line)
            dict_lists.append(d)

c = Counter()
for d in dict_lists:
    c.update(d)
finalDict=dict(c)

dd=pd.DataFrame.from_dict(finalDict,orient='index')
dd.reset_index(inplace=True)
dd = dd.rename(columns={"index":"concept",0:"Frequency_across_Pubmed"})
print(dd)
dd.to_csv("Datasets/Concept_Frequency_across_Pubmed_Abstract.csv")