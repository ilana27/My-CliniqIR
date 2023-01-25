# CliniqIR: Unsupervised Diagnostic Decision Support via Information Retrieval

This is the repository for the paper Unsupervised Diagnostic Decision Support via Information Retrieval

## Datasets
1. Download [PubMed Abstracts](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/).

  ```
  cd CliniqIR_model
  rsync -Pav ftp.ncbi.nlm.nih.gov::pubmed/baseline/\*.xml.gz Pubmed/
  ```
  
2. Download [MIMIC-III datasets](https://mimic.mit.edu/docs/gettingstarted/).

3. Download [DC3 datasets](https://github.com/codiag-public/dc3).

## Requirements
1. Install [QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS)
2. jdk-17.0.2
3. [requirements.txt](https://github.com/rsinghlab/CliniqIR/blob/efef52f87a7fba8faa4a05f209f85ef6daf08fec/requirements.txt)

## Data Preprocessing
Each model requires input data to be in a certain format. Sample files have been provided in the Datasets and CliniqIR_model folder. See/Run the Data_Pre_processing_MIMIC_III.py for MIMIC-III data pre-processing.

## Use CliniqIR 
The index has four fields: pmid, UMLS concepts of an abstract, abstract title and abstract text with the latter two searchable. The source java files have also been provided to allow for custom use.

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
1. Filter text queries by running QuickUMLS_FIltering.py in the Data_preprocessing directory.
2. Save filtered queries in the directory "CliniqIR_model/Queries.txt"
3. Search the index 
```
  cd CliniqIR_model
  java -jar Search_Pubmed_Index.jar -cp LuceneJARFiles2
```
### Evaluate CliniqIR and obtain ensemble results for MIMIC-III.
1. Calculate the PubMed collection frequency of each disease class label by running PubMed_Frequency.py in the Data preprocessing directory.
2. Get Clinical BERT's ranks by running Clinical_BERT.py which can be found in the Bert_models directory. 
3. Obtain CliniqIR's query results by searching the PubMed index.
4. Obtain CliniqIR's ranks and get ensemble results by running Evaluate_Mimic-III.py which can be found in the CliniqIR_model directory.

### To use other Clinical BERT or the zero shot baselines
1. Run Clinical_BERT.py to use Clinical BERT
2. Run Zero_shot_baselines.py to use the zero-shot baselines.



## Credits
Some of the structure in this repo was adopted from https://github.com/ziy/medline-indexer

## Authors
Tassallah Amina Abdullahi

## Reference
1. Luca Soldaini and Nazli Goharian. "QuickUMLS: a fast, unsupervised approach for medical concept extraction." MedIR Workshop, SIGIR 2016.

2. Eickhoff, Carsten, et al. "DC3--A Diagnostic Case Challenge Collection for Clinical Decision Support." Proceedings of the 2019 ACM SIGIR International Conference on Theory of Information Retrieval. 2019.
