# From results to conclusions

## Introduction
Ability for abstract thinking is arguably the most important feature of human intelligence. Scientific domain is the field where this ability is pronounced and scaled.  Drawing conclusions from results conditioning on aims and methods is where abstract reasoning is most pronounced. Also it is the most critical part of scientific process. Publication abstracts are the most compressed form of scientific results and conclusions. Furthermore, scientific abstracts often  have explicit structure with sections like "aim", "method", "result", "conclusion". We parse the  structured abstracts to create a benchmark dataset for evaluating abstract reasoning capabilities of AI models. The PubMed database provide information about publication types and biomedical sub-fields that can be used for generating specific datasets. For example we evaluated clinical case reports and observational study abstracts separately. Another advantage of this approach is that PubMed database has millions of abstracts with daily updates having ~1000 additional abstracts with the result-conclusion structure.
## Methods
When parsing PubMed abstracts is relatively straightforward(see below), evaluating the LLM generated conclusions is a challenge. A two-stage evaluation with multiple strong LLMs as "judges" is used.  The code flow:
- **Generate conclusion** from  context sections
- **Evaluate response** with multiple LLMs("judges") using the real conclusion section from the abstract as reference
- **Ensemble evaluations** to produce a score
The LMM evaluation consists of following steps: 
- "completeness" where LLM judges check for omissions comparing to the reference conclusion statement . The LLM labels are mapped to numeric scores as:
    "No Omissions": 1.0,
    "Minor Omission(s)": 0.5,
    "Major Omission(s)": 0.0,
    "Invalid": None,
    "unknown": 0

- "grounding" where LLM judges check if each generated sentence is supported by the result statement. The LLM labels are mapped to numeric scores as:
    "not_supported": 0.0,
    "supported": 1.0,
    "no_rad": None,
    "unknown": 0

- Multiple judges are averaged
 
The LLM generation and evaluation use the [Kaggle benchmark library](https://github.com/Kaggle/kaggle-benchmarks). Thus the code is portable to Kaggle Benchmark where free credits are provided for running LLM evaluations. Many code elements are borrowed from the [FACTS Grounding benchmark](https://www.kaggle.com/code/prathameshbang/facts-grounding-v2-benchmark-starter). 
## Results
The final model score is the mean of 100 random abstract sample scores. Each sample score is average of "completeness" and "grounding" scores. The sample "grounding" score is average of sentence "grounding" scores. The model scores are available in the [Kaggle Benchmark](https://www.kaggle.com/benchmarks/vmorozov/conclusions-from-results). The individual  scores are assembled into one [table](suppl/all_scores.csv). Beware the CSV table scores are different from the published Kaggle Benchmark scores because they are calculated with ./scripts/scoring.py and ./util/bench.py using  different mapping ("unknown" is mapped to None, instead of 0), aggregation that ignore missing scores and empty responses to deal with over-budget API errors, and includes LLM scores for LLMs that are not included into the published Kaggle benchmark. The individual scores are analyzed with mixed-effects model where the PubMed abstract ids are random effects and LLM models, judge models, metric type(completeness/grounding), length of the response(i.e. generated conclusion) are fixed effects. The results are available in the [notebook](./notebooks/post_analysis.ipynb). As one might expect, "completeness" and "grounding"  are conflicting metrics. Longer responses are better on "completeness" metric, while shorter responses are better on "grounding" metric. Interaction effects between response length, metric type and LLM models are significant (see [line plots](./suppl/score_vs_length.png)). It is particularly pronounced for Anthropic Opus and Sonnet 4.6 models. These models generate [longer conclusions than other models](./suppl/length_distribution.png). Looking into the more difficult clinical case abstracts (lower scores) we see the model "grounding" deteriorates with response length increase. Overall the Anthropic models are high on "completeness" and low on "grounding".  Taking together these observations warrant temperature decrease in the model API for this particular task. Looking into the linear mixed-effect model coefficients and line plots one can see that GPT-5.4-nano and Claude Opus 4.6 are significantly more "lenient" judges than Gemini-3.1 Pro. [Plotting "judge" and metric scores against each other](suppl/judge_vs_metric_scores.png) show some correlation between judges inside same metric. However these correlation are pretty low, so multiple judges are still warranted
# HowTo
## PubMed dataset construction
You might need to have  NCBI account and  API key. Then put them into 'NCBI_EMAIL' and 'NCBI_API_KEY' environment variables or .env file 
The result dataset CSV table will have columns: PMID, DOI, publication type, MeSH terms, and parsed sections of the abstract: aim,method,result,conclusion.

You might find useful documents in read/ folder to understand the search queries. In following examples numbers of retrieved PubMed records are kept around 1000 by adjusting publication date range("crdt" field).
### Using with 'fire' module
```bash
python -m fire scripts/abstract.py --help
python -m fire scripts/abstract.py fetch_pubmed_data_to_file --query=<PubMed query> --output_filename='data/<name>.json'
python -m fire scripts/abstract.py  pubmed2kaggle --name=<name> --dir_data='data' --description="<description>" --title="<title>"
```
### General PubMed dataset
```bash
python -m fire scripts/abstract.py fetch_pubmed_data_to_file --query='(conclusion*[Title/Abstract]) AND 2026/02/01:2026/02/02[crdt] NOT (1800:2025[epdat] OR 1800:2025[edat])' --output_filename='data/pubmed.json'

python -m fire scripts/abstract.py  pubmed2kaggle --name='pubmed' --dir_data='data' --description="Extracted conclusions and results from pubmed abstracts" --title="Abstract Conclusions and Results - pubmed"
```
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
## Benchmark Task notebooks
The Kaggle Benchmark require Python code submitted via notebook. Find examples in notebooks/<name>-conclusion folders. All notebooks share same 