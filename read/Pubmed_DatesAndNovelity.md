# **Temporal Isolation and Metadata Architecture: A Comprehensive Analysis of NCBI PubMed Date Fields for the Prevention of Large Language Model Data Contamination**

The convergence of high-throughput biomedical publishing and the evolution of massive generative AI architectures has created a significant methodological crisis in clinical research. As large language models (LLMs) are increasingly integrated into diagnostic support, evidence synthesis, and radiological interpretation, the integrity of these systems depends fundamentally on the "unseen" nature of the data used for their evaluation. In this context, the National Center for Biotechnology Information (NCBI) PubMed database serves not merely as a repository of scientific knowledge, but as a complex temporal system whose metadata can be leveraged to establish rigorous boundaries between the parametric memory of AI models and novel human discoveries. To navigate this landscape, researchers must move beyond a superficial understanding of publication years and engage with the granular architecture of database ingestion, indexing cycles, and the mechanisms by which web-scale archives like Common Crawl metabolize scientific literature into training sets.

## **The Structural Evolution of PubMed Date Fields**

The National Library of Medicine (NLM) has refined the temporal metadata of its databases over decades to accommodate shifts from physical print cycles to instantaneous electronic dissemination. Understanding which date field signifies public availability requires an examination of the various timestamps assigned to a record during its lifecycle from a publisher's submission to its final archival state.

### **Primary Searchable Date Metadata**

At the core of the PubMed search interface are several distinct date tags, each serving a specific administrative or chronological function. The most visible is the Date of Publication \[dp\], which is the official date assigned by the journal. This field is inherently complex because it encapsulates multiple publishing models, including print, electronic-only, and electronic-ahead-of-print.1 For records following the "Print-Electronic" model, the \[dp\] field reflects the print date, but the electronic date of publication is often earlier.2 Conversely, in "Electronic-Print" models, the electronic date is the primary citation date. This ambiguity makes \[dp\] an unreliable metric for researchers seeking to isolate data from LLM training sets, as a "2025" publication date might represent an abstract that was available online and crawled in mid-2024.1

A more technically precise field is the Entrez Date \[edat\], which historically represents the date the record was first added to the PubMed database.1 For the vast majority of contemporary citations, \[edat\] is the "publicly available" timestamp. It determines the default "Most Recent" sort order and serves as the trigger for My NCBI saved search updates and RSS feeds.1 However, the NLM maintains a policy for older citations where the \[edat\] is retrospectively modified. Since December 2008, if a record enters PubMed more than twelve months after its publication date, the \[edat\] is set equal to the \[dp\] to prevent historical gaps from appearing as new discoveries in automated alerts.4

To provide an immutable record of ingestion, the NLM introduced the Create Date \[crdt\] in late 2008\. The \[crdt\] is the absolute date the citation record was first created in the PubMed system and is never modified based on the publication year.7 For records added prior to its implementation, the \[crdt\] was set equal to the \[edat\].7 For all subsequent records, \[crdt\] provides the most reliable indicator of exactly when a piece of text—including the abstract—became accessible to both human researchers and automated web scrapers.1

| Field Tag | Field Name | NLM Technical Definition | Functional Utility for AI Research |
| :---- | :---- | :---- | :---- |
| \[dp\] | Date of Publication | Date supplied by the publisher; covers both print and electronic versions.1 | High-level chronological filtering; unreliable for ingestion tracking.1 |
| \[edat\] | Entrez Date | The date the citation entered the PubMed/Entrez system.6 | Primary sort order; prone to retrospective modification for older records.4 |
| \[crdt\] | Create Date | The date the PubMed record was first created.3 | The "gold standard" for identifying the date of public availability and potential scraping.7 |
| \[epdat\] | Electronic Date | The date of electronic publication (if applicable).3 | Critical for identifying "Ahead of Print" abstracts that appear before the formal \[dp\].10 |
| \[ppdat\] | Print Date | The date the article appeared in a print issue.1 | Useful for historical bibliometric analysis of physical distribution.1 |
| \[mhda\] | MeSH Date | The date the citation was indexed with MeSH terms.1 | Indicates the record has completed full processing and human/automated curation.12 |

### **Advanced Metadata and Internal Processing**

Beyond the primary searchable fields, the PubMed XML schema includes internal processing dates that offer deeper insights into the bibliographic workflow. The Date Created (DA) field (not to be confused with \[crdt\]) indicates when the Data Creation and Maintenance System (DCMS) first began processing a record.12 The Date Completed (DCOM) indicates when this process was finished. These fields are typically visible only in tagged formats like MEDLINE or XML.12 Furthermore, the Date Revised \[lr\] tracks any subsequent modifications to the record, such as the addition of a previously missing abstract or the correction of author affiliations.12

The existence of the \[lr\] field is particularly relevant for LLM contamination studies. A record might have a \[crdt\] in 2024, but if the abstract was not supplied until 2025, the model's training data would not have contained that text during an early-2025 crawl.12 However, the most conservative approach for researchers is to assume that once a record exists in PubMed, its abstract is either present or imminent, and thus susceptible to ingestion by AI training pipelines.

## **The Ingestion Engine: From Publisher SFTP to Public Searchability**

The mechanism by which an abstract becomes publicly available is a highly automated, near-real-time process. Publishers of journals indexed in PubMed are required to submit citation and abstract data electronically in a specific XML format.14 These files are uploaded via Secure File Transfer Protocol (SFTP). The NLM's loader scripts are configured to monitor these SFTP directories at ten-minute intervals, 24 hours a day, seven days a week.14

Once a properly formatted file is uploaded, the following timeline generally applies:

1. **Immediate Intake**: The system generates a loader report within an hour, confirming receipt of the citation.14  
2. **Indexing Initiation**: The record is assigned a unique PubMed Identifier (PMID). The \[crdt\] is set based on this initial creation.7  
3. **Public Searchability**: New and updated records usually appear in the public PubMed interface within 24 hours of successful intake.10  
4. **Automated MeSH Indexing**: Since April 2022, the NLM has transitioned to a fully automated indexing system. For MEDLINE-scope journals, citations are typically indexed with MeSH terms within 24 hours of their appearance in PubMed, a significant reduction from the previous one-to-three-month lag.13  
5. **Human Quality Assurance**: A subset of these automatically indexed citations undergoes human review for quality assurance, usually within two weeks of entering the database.13

This rapid cycle means that an abstract published in a journal like the *Journal of Medical Internet Research* (JMIR) can be searchable in PubMed within 24 to 48 hours of final acceptance.10 For models that leverage live-retrieval tools, this window of "unseen" data is practically non-existent. However, for the "parametric knowledge" stored in the model's weights, the crucial factor is not just when the data appeared in PubMed, but when the next training "snapshot" was taken by LLM developers.

## **The LLM Training Pipeline and Data Contamination**

To avoid using abstracts that have been utilized for training LLMs, one must understand the provenance of the text data used by organizations like OpenAI, Anthropic, and Google. These entities rarely crawl the web in real-time for pre-training. Instead, they rely on massive, curated corpora such as Common Crawl, the Colossal Cleaned Corpus (C4), and specialized collections like The Pile.17

### **The Role of Common Crawl and Harmonic Centrality**

Common Crawl is a multi-petabyte archive of the web that serves as the backbone for most major LLMs. For instance, over 80% of the training tokens for GPT-3 were derived from filtered Common Crawl archives.18 Common Crawl releases new "snapshots" of the internet monthly. However, it does not crawl every page on every site with the same frequency. It utilizes a metric known as Harmonic Centrality (HC) to prioritize which domains to crawl.19

Harmonic Centrality measures how "close" a domain is to all other domains in the global web graph.20 Domains with high authority and extensive inbound links—such as pubmed.ncbi.nlm.nih.gov—possess extremely high HC scores. Consequently, Common Crawl visits these sites more frequently, often multiple times within a single month's snapshot cycle.19 This means that a PubMed abstract has a high probability of being captured by Common Crawl almost immediately after it becomes available. If an abstract appears in September 2025, it will likely be included in the September or October 2025 Common Crawl release.19

### **From Crawl to Weights: The Processing Lag**

While the data is captured by Common Crawl quickly, there is a substantial lag before it is integrated into a model's weights. Training a frontier model like GPT-5 or Claude 4 requires months of computational work.21 Once training is finished, the model undergoes weeks of safety testing and alignment (RLHF) before public release.21 The "knowledge cutoff date" is the timestamp of the last data point included in the pre-training set.22

| Model Series | Specific Architecture | Public Release Date | Reliable Knowledge Cutoff |
| :---- | :---- | :---- | :---- |
| **OpenAI GPT** | GPT-5.2 | December 2025 | August 2025 23 |
|  | GPT-5.3 Codex | February 2026 | October 2025 23 |
|  | GPT-o1 (o1) | September 2024 | October 2023 23 |
|  | GPT-4o | May 2024 | October 2023 23 |
| **Anthropic Claude** | Claude 4.6 Opus | February 2026 | May 2025 23 |
|  | Claude 4.5 Opus | November 2025 | March 2025 23 |
|  | Claude 3.7 Sonnet | February 2025 | November 2024 24 |
|  | Claude 3.5 Sonnet | June 2024 | April 2024 23 |
| **Google Gemini** | Gemini 3.1 Pro | February 2026 | January 2025 23 |
|  | Gemini 3.0 Pro | November 2025 | January 2025 23 |
|  | Gemini 2.5 Pro | March 2025 | November 2024 24 |
|  | Gemini 2.0 Flash | December 2024 | August 2024 23 |
| **Meta Llama** | Llama 4 Scout | April 2025 | August 2024 25 |
|  | Llama 3.1 (405B) | July 2024 | December 2023 23 |
| **DeepSeek** | DeepSeek-V3 | December 2024 | December 2024 23 |
|  | DeepSeek-R1 | January 2025 | January 2025 24 |

This table illustrates the critical "temporal safety window" for researchers. To ensure that a model like GPT-5.2 has never seen a specific abstract in its training data, the abstract's PubMed \[crdt\] should post-date August 2025\.23 However, for models like DeepSeek-R1, which are known for rapid iteration and "real-time" data ingestion, the cutoff may be as recent as the month of release.25

## **The Benchmark Crisis: Data Contamination in Medical AI**

The primary reason for identifying "unseen" abstracts is the pervasive issue of data contamination in medical AI evaluation. Traditional benchmarks such as PubMedQA or MedQA rely on static datasets. If these datasets were published in 2020, they have been crawled and "memorized" by every modern LLM.26 When a model achieves 95% accuracy on these tests, it is often impossible to determine if the model is "reasoning" or simply performing "parametric recall"—retrieving the answer from its internal memory rather than analyzing the provided text.26

### **Evaluating Knowledge Discovery Ability**

Rigorous evaluation of an AI's capacity for genuine knowledge discovery requires "temporal separation." This means the evaluation data must post-date the model's release or training cutoff to ensure the information was not seen during training.26 When models are tested on events that post-date their cutoff, they are said to be operating under "True Ignorance" (TI). Conversely, if they are asked to "forget" what they know about historical events, they are in a state of "Simulated Ignorance" (SI), which is significantly easier for the model to bypass.29

Recent research in mathematical biology and rare diseases highlights this gap. While models like Gemini 3 Pro excel at retrieval-augmented tasks, their performance on genuine "discovery" benchmarks (extracting novel QA pairs from post-cutoff literature) remains markedly lower than on pre-cutoff data.26 This suggests that the apparent "superhuman" performance of LLMs in medical exams may be an artifact of contamination.

### **Statistical Frameworks for "Unseen" Knowledge**

To quantify what a model truly "knows" versus what it "recalls," researchers have proposed statistical frameworks like KnowSum. This approach treats knowledge as an "unseen species" problem in ecology. By querying a model multiple times with the same prompt and observing the frequency of unique responses, one can use the Good–Turing (GT) estimator to predict how much knowledge remains undiscovered by the model.28

The Smoothed Good–Turing (SGT) estimator for unseen capacity is defined as:

![][image1]  
where ![][image2] is the number of items that appear exactly ![][image3] times in the first ![][image4] observations, and ![][image5] is the extrapolation factor.28 This mathematical approach allows researchers to evaluate the depth of a model's internal knowledge without needing a clean external dataset, though the gold standard remains the use of temporally isolated PubMed abstracts.

## **Practical Search Strategies for Temporal Isolation**

For a researcher seeking to find the most recent publicly available abstracts to avoid training contamination, the following search methodologies are recommended.

### **Building Web Queries in PubMed**

The most direct way to find post-cutoff abstracts is to use the \[crdt\] tag in the PubMed search bar. If the goal is to evaluate a model with a January 2025 cutoff, the query should target records created after February 1, 2025\.

**Standard Query Template:**

"clinical concept" AND 2025/02/01:3000\[crdt\]

In this syntax, 3000 is a placeholder for the present date, ensuring all records from the start point forward are included.1 To ensure the results include abstracts (as many new citations initially enter PubMed as "metadata-only" records), the researcher should apply the "has abstract" filter.31

**Refined Query with Abstract Filter:**

"lung cancer therapy" AND hasabstract AND 2025/03/01:3000\[crdt\]

### **Utilizing EDirect for Automated Retrieval**

For large-scale dataset construction, the NLM's E-Utilities (EDirect) provide a command-line interface for complex temporal filtering. The esearch tool allows for the specification of a \-datetype and a range.11

**EDirect Example for Recent Ingestion:**

esearch \-db pubmed \-query "alzheimers" \-datetype CRDT \-mindate 2025/05/01 \-maxdate 2026/04/01 | efetch \-format abstract \> recent\_abstracts.txt

This command specifically targets the Create Date.11 If a researcher wants to find the most recently *added* papers (regardless of their publication date), they can use the \-days argument:

esearch \-db pubmed \-query "cardiology" \-datetype EDAT \-days 30 11

This retrieves all papers that entered the Entrez system in the last 30 days.

### **Filtering by Publication Type and Subset**

To further ensure the quality and "newness" of the data, researchers can use the Subset \[sb\] tag. This allows for the restriction of results to specific citation statuses or categories.15

**Excluding Preprints and Older Subsets:** "gene therapy" AND 2025/06/01:3000\[crdt\] NOT preprint\[pt\] 34

Preprints are particularly risky for contamination because they are often indexed in multiple servers (bioRxiv, medRxiv, PMC) and are highly visible to web crawlers long before their final PubMed \[crdt\] is established.35

## **Historical Context and Legacy Policies**

The transition of the Entrez Date \[edat\] from a simple ingestion timestamp to a managed field is a critical historical detail for researchers working with older datasets. Prior to September 1997, the NLM did not have a robust system for tracking ingestion dates separately from publication dates.4 In late 2008, the NLM changed its policy so that citations entering the database more than a year after publication would have their \[edat\] set to the publication date.4

This policy change was designed to preserve the relevance of "What's New" searches. For example, if NLM decided to index a historical journal from 1950 in the year 2026, setting the \[edat\] to 2026 would cause thousands of 1950s papers to flood the "Recently Added" feeds of current researchers.4 By setting \[edat\] to 1950, these records remain buried in historical searches. For the AI researcher, this means that \[edat\] *cannot* be used to find when a historical abstract was first made available online if it was added to PubMed significantly after its print publication.4 In these specific cases, \[crdt\] is the only field that accurately reflects the 2026 ingestion.7

## **Identifying "Ahead of Print" Lag Times**

A significant portion of modern scientific literature is released as "Electronic Publication Ahead of Print" (Epub). In PubMed, these records are identified by the \[epdat\] tag.3 Lag times between online availability and official print publication vary significantly. A survey of 1,245 network meta-analyses (NMAs) found that the median time from the last systematic search to publication was 11.6 months, with the median "submission lag time" (the time between search and submission) being 6.8 months.37

From the perspective of LLM training, an Epub record is "publicly available" as soon as it appears in PubMed, even if the formal \[dp\] is months in the future. Journals often submit metadata to PubMed the same day an article is published online.16 Consequently, researchers must use the \[epdat\] or \[crdt\] fields to catch these early-release abstracts that might otherwise be missed if one only searches by publication year.3

## **The Future of AI-Resistant Data Retrieval**

As LLM providers move toward more frequent "fine-tuning" on recent web data, the concept of a "clean" dataset becomes a moving target. OpenAI's GPT-5.2 was released merely one month after GPT-5.1, suggesting that training pipelines are becoming more agile.26

### **Strategic Recommendations for Researchers**

1. **Monitor "Reliable" vs "Parametric" Cutoffs**: Organizations like Hatz AI or LLMpulse distinguish between a model's "reliable" knowledge (data deeply integrated into weights) and its "parametric" cutoff (the absolute latest data it might have seen).21 Always target the parametric cutoff for evaluation.  
2. **Verify via PRISM**: Use correlation-based tests like PRISM to detect if a chosen set of PubMed abstracts was likely part of a model's training corpus.38  
3. **Timestamp definitively**: When publishing new medical AI benchmarks, authors should include clear timestamps and the exact PubMed query (including \[crdt\] ranges) used to construct the dataset.22  
4. **Leverage the \[lr\] (Date Revised) field**: For stability testing, use records with recent \[lr\] dates to identify content that may have been corrected or expanded *after* an initial training crawl.12

The ability to delineate when an abstract became publicly available is not just a technical requirement for effective searching; it is a fundamental pillar of the scientific method in the age of artificial intelligence. By utilizing the \[crdt\] field in conjunction with a deep understanding of LLM training cycles and web-archival priorities, researchers can continue to produce rigorous, unbiased evaluations of AI performance in the biomedical domain.

## **Detailed Comparative Analysis of PubMed Date Metadata**

| Feature | Date of Publication \[dp\] | Entrez Date \[edat\] | Create Date \[crdt\] | Date Revised \[lr\] |
| :---- | :---- | :---- | :---- | :---- |
| **Search Tag** | \[dp\] | \[edat\] | \[crdt\] | \[lr\] |
| **Stability** | May change if print/electronic years differ.1 | May be changed for older records.4 | Immutable once created.7 | Changes with any record update.12 |
| **Default Sort** | Optional | Yes ("Most Recent").1 | No | No |
| **Publicly Available?** | Not necessarily (print may lag).1 | Generally yes, at the time of entry.4 | Yes, the absolute start of PubMed life.7 | Yes, reflects the latest version.12 |
| **Searchable?** | Yes | Yes | Yes (since Dec 2008).7 | Yes (on tagged formats).12 |
| **Rangeable?** | Yes | Yes | Yes | Yes (in EDirect).11 |

By strictly adhering to the \[crdt\] and \[edat\] fields, and avoiding the misleading official \[dp\], researchers can successfully navigate the complexities of the NCBI metadata ecosystem and maintain the temporal isolation required for state-of-the-art AI research.

# Tips
To identify the most recent PubMed abstracts while specifically excluding those that appeared as preprints before 2025, you must move beyond filtering by **Publication Type** and instead target the metadata that links journal articles to their previous versions.

### The Problem: Why `NOT preprint[pt]` is Insufficient
The `[pt]` (Publication Type) tag only identifies the **record type**. PubMed treats a preprint and its subsequent journal article as two **separate records** with different PMIDs.[1, 2]
*   **The Preprint Record:** Has `preprint[pt]`. Adding `NOT preprint[pt]` successfully hides this record.
*   **The Journal Article Record:** Has `journal article[pt]`. It will still appear in your results even if its content was identical to a 2024 preprint that the LLM has already "seen" during training .

### The Solution: Filtering the "Update" Relationship
When a journal article is published that was previously a preprint, the NLM often creates a metadata link between them. In the journal record, this appears in the **Comments/Corrections** field as an **"Update of"** relationship .

#### 1. Exclude the "Update of" Metadata
You can exclude journal articles that are explicitly identified as updates to a previous version (usually a preprint) by searching against all fields for the relationship tag.
*   **Syntax:** `NOT "update of"[all]`
*   *Note:* While this also excludes legitimate updates to systematic reviews, it is the most effective way to ensure you are looking at "Version 1" research that has no prior existence in the database .

#### 2. Exclude Preprint Server Names
Some records may not have the formal "Update of" link yet but will mention the preprint server in the metadata or affiliation.
*   **Syntax:** `NOT biorxiv[all] NOT medrxiv[all] NOT researchsquare[all] NOT arxiv[all]`

#### 3. Use the Electronic Publication Date (`[epdat]`)
Many articles are published "Ahead of Print" (Epub). If a paper has a 2025 journal date but was an Epub in late 2024, it was likely crawled by LLM training sets like Common Crawl.[3, 4, 5]
*   **Syntax:** `NOT 1800:2024[epdat]`
*   This ensures the very first electronic version of the journal article also appeared in 2025 or later.[6, 7]

### The Recommended "Clean Data" Search Query
To find abstracts added since January 2025 that were likely never available in preprint or electronic form before that date, use this combined string:

"your search terms" AND 2025/01/01:3000[crdt] AND hasabstract NOT preprint[pt] NOT 1800:2024[epdat] NOT "update of"[all] NOT biorxiv[all] NOT medrxiv[all]

### Breakdown of the Search Tags

| Tag Component | Purpose |
| :--- | :--- |
| `2025/01/01:3000[crdt]` | Targets the **Create Date**. Ensures the record first entered the PubMed system in 2025 . |
| `hasabstract` | Filters out metadata-only records, ensuring you only retrieve entries with text.[8] |
| `NOT preprint[pt]` | Excludes the standalone preprint records.[9, 10] |
| `NOT 1800:2024[epdat]` | Excludes "Ahead of Print" articles that were available electronically in 2024.[6, 11] |
| `NOT "update of"[all]` | Excludes journal articles that PubMed has flagged as being a new version of an existing preprint . |
| `NOT biorxiv[all]` | Secondary safety net to catch any mention of the most common medical preprint servers.[12] |

### Statistical Verification (Advanced)
If you are building a benchmark for LLM evaluation, the most conservative approach is to use the **Create Date** with a 2-3 month buffer after the LLM's known cutoff.[13, 14] For a model with a "January 2025" cutoff, targeting records with a `[crdt]` of **March 2025 or later** is safer, as it accounts for the lag between an abstract appearing on the web and its integration into a model's parametric memory.[15, 16]

#### **Works cited**

1. Help \- PubMed \- NIH, accessed April 8, 2026, [https://pubmed.ncbi.nlm.nih.gov/help/](https://pubmed.ncbi.nlm.nih.gov/help/)  
2. Use of Article and ArticleDate Attribute Values in Creating the Source Area of the MEDLINE/PubMed Citation Display \- National Library of Medicine, accessed April 8, 2026, [https://www.nlm.nih.gov/bsd/licensee/elements\_article\_source.html](https://www.nlm.nih.gov/bsd/licensee/elements_article_source.html)  
3. PubMed User Guide, accessed April 8, 2026, [https://webmed.irkutsk.ru/doc/pdf/pubmed.pdf](https://webmed.irkutsk.ru/doc/pdf/pubmed.pdf)  
4. PubMed Entrez Date Modification for Older Citations \- National Library of Medicine \- NIH, accessed April 8, 2026, [https://www.nlm.nih.gov/pubs/techbull/so08/so08\_pm\_edat.html](https://www.nlm.nih.gov/pubs/techbull/so08/so08_pm_edat.html)  
5. Pubmed User Guide | PDF | Pub Med | Information Science \- Scribd, accessed April 8, 2026, [https://www.scribd.com/document/512680590/pubmed](https://www.scribd.com/document/512680590/pubmed)  
6. Why Citations to Older Articles May Display Before More Recent Ones in PubMed, accessed April 8, 2026, [https://www.nlm.nih.gov/pubs/techbull/ma02/ma02\_display\_order\_beta.html](https://www.nlm.nih.gov/pubs/techbull/ma02/ma02_display_order_beta.html)  
7. Create Date — New Field Indicates When Record Added to PubMed, accessed April 8, 2026, [https://www.nlm.nih.gov/pubs/techbull/nd08/nd08\_pm\_new\_date\_field.html](https://www.nlm.nih.gov/pubs/techbull/nd08/nd08_pm_new_date_field.html)  
8. ZD410 Desktop Printer Support & Downloads \- Hayawin, accessed April 8, 2026, [https://www.hayawin.com/resources/zd410-desktop-printer-support-amp-downloads.html](https://www.hayawin.com/resources/zd410-desktop-printer-support-amp-downloads.html)  
9. PubMed Entrez Date Modification for Older Citations. NLM Technical Bulletin. 2008 Sep–Oct, accessed April 8, 2026, [https://www.nlm.nih.gov/pubs/techbull/so08/so08\_pm\_edat\_beta.html](https://www.nlm.nih.gov/pubs/techbull/so08/so08_pm_edat_beta.html)  
10. What is the "PubMed Now\!" ("ahead-of-print") option when I pay the ..., accessed April 8, 2026, [https://support.jmir.org/hc/en-us/articles/360008899632-What-is-the-PubMed-Now-ahead-of-print-option-when-I-pay-the-APF](https://support.jmir.org/hc/en-us/articles/360008899632-What-is-the-PubMed-Now-ahead-of-print-option-when-I-pay-the-APF)  
11. esearch \- The Insider's Guide to Accessing NLM Data \- National Library of Medicine, accessed April 8, 2026, [https://dataguide.nlm.nih.gov/edirect/esearch.html](https://dataguide.nlm.nih.gov/edirect/esearch.html)  
12. Changes to PubMed for 2001 \- National Library of Medicine \- NIH, accessed April 8, 2026, [https://www.nlm.nih.gov/pubs/techbull/jf01/jf01\_pubmed\_2001\_beta.html](https://www.nlm.nih.gov/pubs/techbull/jf01/jf01_pubmed_2001_beta.html)  
13. NLM Office Hours: PubMed \- National Library of Medicine \- NIH, accessed April 8, 2026, [https://www.nlm.nih.gov/oet/ed/pubmed/06-24\_oh-pubmed.html](https://www.nlm.nih.gov/oet/ed/pubmed/06-24_oh-pubmed.html)  
14. XML Help for PubMed Data Providers \- NCBI \- NIH, accessed April 8, 2026, [https://www.ncbi.nlm.nih.gov/books/NBK3828/](https://www.ncbi.nlm.nih.gov/books/NBK3828/)  
15. PubMed features to save your time \- PMC \- NIH, accessed April 8, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11029388/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11029388/)  
16. How long after publication does it take for an article to show up in PubMed?, accessed April 8, 2026, [https://support.jmir.org/hc/en-us/articles/115001813331-How-long-after-publication-does-it-take-for-an-article-to-show-up-in-PubMed](https://support.jmir.org/hc/en-us/articles/115001813331-How-long-after-publication-does-it-take-for-an-article-to-show-up-in-PubMed)  
17. The Top 10 LLM Training Datasets for 2026 \- Open Data Science, accessed April 8, 2026, [https://opendatascience.com/the-top-10-llm-training-datasets-for-2026/](https://opendatascience.com/the-top-10-llm-training-datasets-for-2026/)  
18. A Critical Analysis of the Largest Source for Generative AI Training Data: Common Crawl \- ACM FAccT, accessed April 8, 2026, [https://facctconference.org/static/papers24/facct24-148.pdf](https://facctconference.org/static/papers24/facct24-148.pdf)  
19. Common Crawl Harmonic Centrality AI Optimization \- LLMrefs, accessed April 8, 2026, [https://llmrefs.com/blog/common-crawl-harmonic-centrality-ai-optimization](https://llmrefs.com/blog/common-crawl-harmonic-centrality-ai-optimization)  
20. How SEOs Are Using Common Crawl's Web Graph Data for AI Ranking Signals, accessed April 8, 2026, [https://commoncrawl.org/blog/how-seos-are-using-common-crawls-web-graph-data-for-ai-ranking-signals](https://commoncrawl.org/blog/how-seos-are-using-common-crawls-web-graph-data-for-ai-ranking-signals)  
21. Understanding LLM Training Dates \- Hatz AI Help Center, accessed April 8, 2026, [https://docs.hatz.ai/en/articles/11961691-understanding-llm-training-dates](https://docs.hatz.ai/en/articles/11961691-understanding-llm-training-dates)  
22. Knowledge Cutoff in AI: what it is and how models handle it \- LLM Pulse, accessed April 8, 2026, [https://llmpulse.ai/blog/glossary/knowledge-cutoff-in-ai/](https://llmpulse.ai/blog/glossary/knowledge-cutoff-in-ai/)  
23. Knowledge Cutoff Date: Definition & Explanation \- Gradually AI, accessed April 8, 2026, [https://www.gradually.ai/en/ai-glossary/knowledge-cutoff-date/](https://www.gradually.ai/en/ai-glossary/knowledge-cutoff-date/)  
24. LLM Leaderboard 2025 \- Vellum AI, accessed April 8, 2026, [https://www.vellum.ai/llm-leaderboard](https://www.vellum.ai/llm-leaderboard)  
25. Top 50+ Large Language Models (LLMs) in 2026 \- Exploding Topics, accessed April 8, 2026, [https://explodingtopics.com/blog/list-of-llms](https://explodingtopics.com/blog/list-of-llms)  
26. Can Large Language Models Derive New Knowledge? A Dynamic Benchmark for Biological Knowledge Discovery \- arXiv, accessed April 8, 2026, [https://arxiv.org/html/2603.03322v1](https://arxiv.org/html/2603.03322v1)  
27. MedMeta: A Benchmark for LLMs in Synthesizing Meta-Analysis Conclusion from Medical Studies | OpenReview, accessed April 8, 2026, [https://openreview.net/forum?id=AwRIQV177K](https://openreview.net/forum?id=AwRIQV177K)  
28. Evaluating the Unseen Capabilities: How Much Do LLMs Actually Know? \- Xiang Li, accessed April 8, 2026, [https://lx10077.github.io/assets/pdf/slides/2025GM\_unseen.pdf](https://lx10077.github.io/assets/pdf/slides/2025GM_unseen.pdf)  
29. Simulated Ignorance Fails: A Systematic Study of LLM Behaviors on Forecasting Problems Before Model Knowledge Cutoff \- arXiv, accessed April 8, 2026, [https://arxiv.org/pdf/2601.13717](https://arxiv.org/pdf/2601.13717)  
30. A systematic assessment of large language models' knowledge of rare diseases \- PMC, accessed April 8, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12796007/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12796007/)  
31. claude-code-templates/cli-tool/components/skills/scientific/pubmed, accessed April 8, 2026, [https://github.com/davila7/claude-code-templates/blob/main/cli-tool/components/skills/scientific/pubmed-database/references/search\_syntax.md?plain=1](https://github.com/davila7/claude-code-templates/blob/main/cli-tool/components/skills/scientific/pubmed-database/references/search_syntax.md?plain=1)  
32. Entrez® Direct: E-utilities on the Unix Command Line \- NCBI, accessed April 8, 2026, [https://www.ncbi.nlm.nih.gov/books/NBK179288/](https://www.ncbi.nlm.nih.gov/books/NBK179288/)  
33. The 9 E-utilities and Associated Parameters \- The Insider's Guide to Accessing NLM Data, accessed April 8, 2026, [https://www.nlm.nih.gov/dataguide/eutilities/utilities.html](https://www.nlm.nih.gov/dataguide/eutilities/utilities.html)  
34. PubMed Update: Article Type filters updated to reflect MeSH changes to Publication Types, accessed April 8, 2026, [https://www.nlm.nih.gov/pubs/techbull/ma26/ma26\_pubmed\_update\_MeSH\_changes.html](https://www.nlm.nih.gov/pubs/techbull/ma26/ma26_pubmed_update_MeSH_changes.html)  
35. NIH Preprint Pilot FAQs \- PMC, accessed April 8, 2026, [https://pmc.ncbi.nlm.nih.gov/about/nihpreprints-faq/](https://pmc.ncbi.nlm.nih.gov/about/nihpreprints-faq/)  
36. Preprints: Facilitating early discovery, access, and feedback \- PMC \- NIH, accessed April 8, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC6191398/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6191398/)  
37. Lag times in the publication of network meta-analyses: a survey \- PMC, accessed April 8, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8422315/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8422315/)  
38. Detecting Non-Membership in LLM Training Data via Rank Correlations \- arXiv, accessed April 8, 2026, [https://arxiv.org/html/2603.22707v1](https://arxiv.org/html/2603.22707v1)

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAAzElEQVR4XmNgGAWjYIiCHCD+D8UgUIrEvwNTRAwwBuI1QMzDANH8F4gZkeRBYpOQ+HjBViDWB2IbBohGMVRpsFgrmhhBcJcB4U0YkIaKCaKJEwTIYQYD84H4H5oYUQBkEEgzDLBAxSyhfFC4wgAoTK8wQOTtkcTBAKYRFBkwEA0VgwFk9k8g5mSAhDW6bzA0goAsVMwKSrMiycGCBCSHAUC2SKILMkAM8EUXBIJgBoSBEWhyJAGQFzWh7KUMFBp2HUozA/FrZIlRQDoAALbPLNN6ykr7AAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAZCAYAAADjRwSLAAAAdElEQVR4XmNgGAU0BbJA/AiIl6JLgAAnEP8HYjMonxuIfyOkIWA5A0QRDBwA4rdIfDB4yABRNAHKZ0aSgwNJBogiZIwTKAPxXwaIIn1kiX9QQWTwFYj5kQVACoqR+MJQMRQwC4hXArEOELsxQBR4o6gY2QAA7DMYlvZyPf4AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAXCAYAAAA/ZK6/AAAAj0lEQVR4XmNgGAWDCTAC8SMg9kYSWwrEd4GYFUkMDr4CsQoQ/wdifijNAcSKULYMQikDQzkQRwOxNFSyGVkSKrYQWQBkLQ8QF0El0QFIDGQgBvjNgKnBD4sYHIAkrmIRa4CyxZDEwQDdak2oGCeUfxpJjsGUAdPqdCSxWcgSIAAyRRJdkAESB77ogqOAEAAAVlIckZLLHCYAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAcAAAAZCAYAAAD9jjQ4AAAAcElEQVR4XmNgGOSAEYh90QVhYCsQ/0cXhAGQBFZJJQaIBEg3HAgAsSQQ10MlI6B8ZmRFv6GSWAFI4gm6IAyAJFvRBUFAnAEiyY0uAQLpDHjs+wbEk5D4KApBHGMomx+IbZHkGDYD8SoGiB9xGj+EAQAs4BaTpEPr1gAAAABJRU5ErkJggg==>