# From results to conclusions

## PubMed datasets
You might need to have  NCBI account and  API key. Then put them into 'NCBI_EMAIL' and 'NCBI_API_KEY' enverinment variables or .env file 
The result dataset CSV table will have columns: PMID, DOI, publication type, MeSH terms, and parsed sections of the abstract: aim,method,result,conclusion.

keep in mind that a Kaggle  dataset title must be between 6 and 50 characters
### Clinical case dataset
```bash
python -m fire scripts/abstract.py fetch_pubmed_data_to_file --query='(Case*[Publication Type]) AND (conclusion*[Title/Abstract]) AND 2026/02/01:2026/03/01[crdt] NOT (1800:2025[epdat] OR 1800:2025[edat])' --output_filename='data/pubmed_case.json'

python -m fire scripts/abstract.py  pubmed2kaggle --name='pubmed_case' --dir_data='data' --description="Extracted case presentation and conclusion from PubMed case report abstracts" --title="PubMed Case Reports with Conclusions"
```

### Observational study dataset
```bash
python -m fire scripts/abstract.py fetch_pubmed_data_to_file --query='(observational study*[Publication Type]) AND (conclusion*[Title/Abstract]) AND 2026/02/01:2026/03/01[crdt] NOT (1800:2025[epdat] OR 1800:2025[edat])' --output_filename='data/pubmed_observation.json'

python -m fire scripts/abstract.py  pubmed2kaggle --name='pubmed_observation' --dir_data='data' --description="Extracted  result and conclusin statments from PubMed observational study abstracts" --title="PubMed Observational Study Conclusions"
```

'pubmed2kaggle' will print kaggle dataset creation command:`kaggle datasets create -p out/result_conclusion/<name>`
## obsolete datasets creation from BibTex

[Run search](https://www.medrxiv.org/search/abstract_title%3Aconclusion%2Bconclusions%20abstract_title_flags%3Amatch-any%20jcode%3Amedrxiv%20limit_from%3A2026-01-01%20limit_to%3A2026-03-15%20numresults%3A75%20sort%3Arelevance-rank%20format_result%3Astandard)
Select by clicking "+Add all citation" on each page. Thes save in the BibTex format

make sure that you have BibTex data/biorxiv.bib or data/medrxiv.bib
Following command parse the bib file to CSV table and create Kaggle dataset manifest using ~/.kaggle/kaggle.json:
```bash
python -m fire scripts/abstract.py bib2kaggle --name medrxiv
```
