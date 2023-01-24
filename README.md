# CliniqIR: Unsupervised Diagnostic Decision Support via Information Retrieval
Code for the paper Unsupervised Diagnostic Decision Support via Information Retrieval


## Datasets
1. Download [PubMed Abstracts](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/).

  ```
  cd CliniqIR_model
  rsync -Pav ftp.ncbi.nlm.nih.gov::pubmed/baseline/\*.xml.gz Pubmed/
  ```
  
2. Download [MIMIC-III datasets](https://mimic.mit.edu/docs/gettingstarted/).

3. Download [DC3 datasets](https://github.com/codiag-public/dc3).

## Requirements
1. [QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS)
2. jdk-17.0.2
3. [requirements.txt](https://github.com/rsinghlab/CliniqIR/blob/efef52f87a7fba8faa4a05f209f85ef6daf08fec/requirements.txt)


## Use CliniqIR 
The index has four fields: pmid, UMLS concepts of an abstract, abstract title and abstract text with the latter two searchable. The source java files are also provided for custom use.

### Building the PubMed Index
1. Download PubMed Abstracts. Abstracts should be in the directory "CliniqIR_model/Pubmed".
2. Extract UMLS Concepts from PubMed Abstracts.

```
  cd Data_Preprocessing
  Python Extract_Pubmed_Concepts.py
```
3. Build the index

```
  cd CliniqIR_model
  java -jar Build_Pubmed_Index.jar -cp LuceneJARFiles2
```
### Searching the PubMed Index

1. Prepare Queries and save it in the directory "CliniqIR_model/Queries.txt"
2. Search the index 
```
  cd CliniqIR_model
  java -jar Search_Pubmed_Index.jar -cp LuceneJARFiles2
```
### Evaluate CliniqIR and obtain Ensemble Results for MIMIC-III.
1. Get Clinical BERT ranks by running Clinical_BERT.py which can be found in the Bert_models directory. 
2. Obtain CliniqIR results by searching the PubMed index.
3. Obtain CliniqIR ranks and get ensemble results by running Evaluate_Mimic-III.py which can be found in the CliniqIR_model directory.

### To use zero shot baselines

## Credits
Some of the structure in this repo was adopted from https://github.com/ziy/medline-indexer

## Authors
Tassallah Amina Abdullahi

## Reference
1. Luca Soldaini and Nazli Goharian. "QuickUMLS: a fast, unsupervised approach for medical concept extraction." MedIR Workshop, SIGIR 2016.

2. Eickhoff, Carsten, et al. "DC3--A Diagnostic Case Challenge Collection for Clinical Decision Support." Proceedings of the 2019 ACM SIGIR International Conference on Theory of Information Retrieval. 2019.
