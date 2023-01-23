import xml as xml
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse
from bs4 import BeautifulSoup
import json
from multiprocessing import Pool
import os, gzip, shutil, glob
from quickumls import QuickUMLS
from collections import defaultdict
from multiprocessing import Pool

"""
This code defines a function called getAbstracts which takes in a file name as an input.
The function reads the xml file, extracts the abstract text, and uses the QuickUMLS matcher 
to extract concepts from the abstract text. It then writes the concepts and their attributes 
in json format to a new file.
It loops through all the files in a directory and extract their UMLS concepts".
"""

# Initialize QuickUMLS matcher
matcher = QuickUMLS(quickumls_fp = "QuickUmlsData")

# Define function to extract files from .gz format
def gz_extract(directory):
    os.chdir(directory)
    for item in os.listdir(directory):
        if item.endswith(".gz"):
            gz_file = os.path.abspath(item)
            file_name = os.path.basename(gz_file).rsplit('.', 1)[0]
            with gzip.open(gz_file, "rb") as f_in, open(file_name, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_file)

#Define function to return list objects instead of set.
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

# Define function to extract concepts from abstracts in xml files
def getAbstracts(file):
    filename = file.split('.')[0]
    outputdir= "CliniqIR_model/Pubmed_Concepts/"
    doc = parse(file).getroot()
    values = [e for e in doc.findall('PubmedArticle/MedlineCitation/Article')]
    newAbstracts=[]
    with open(outputdir+filename+".txt", "w") as f:
        for i in range (len(values)):
            abstract = ET.tostring(values[i]).decode()
            if '<Abstract>'in abstract:
                abstract = abstract.split('<Abstract>')[1].split('</Abstract>')[0]
                soup = BeautifulSoup(abstract)
                soup =soup.get_text()
                matches=matcher.match(soup, best_match=True, ignore_syntax=False)
                result = defaultdict(list)
                for a in range (len(matches)):    
                    term = matches[a][0]["term"]
                    cui = matches[a][0]["cui"]
                    sem = matches[a][0]["semtypes"]
                    result[cui] = [term]
                    result[cui].append(sem)
                    newAbstracts.append(result)
                result = json.dumps(result, default=set_default)
                f.write(str(result)+"\n") 
                
            else:
                result = {}
                result = json.dumps(result)
                f.write((result)+"\n")
            # print("Done with concept extraction")

dir_name = "CliniqIR_model/Pubmed"      
gz_extract(dir_name)
file_list=[]
os.chdir(dir_name)

for files in glob.glob("*.xml"):
        file_list.append(files)
        print("Done with Files")


"""
It then uses the multiprocessing library to execute the getAbstracts function
on all the files in the file_list using the map function of the pool object, 
which makes the process faster.

"""

# if __name__ == '__main__':
#     with Pool(20) as p:
#         p.map(getAbstracts, file_list)
#         p.close() # no more tasks
#         p.join()
#     print("Done with concept extraction")

for i in file_list:
    getAbstracts(i)