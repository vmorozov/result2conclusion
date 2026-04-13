import json
from pybtex.database import parse_string
import pandas as pd
import re
import csv
import re
from Bio import Entrez, Medline
import os
import dotenv
dotenv.load_dotenv()

#parse from data/bibtext.bib and 'doi' and 'abstract' to data/abstract.jsonl    
def parse_abstract3(text):
    introduction = None
    method = None
    results = None
    conclusion = None

    meth_str = r'(?i:Methods)[:.\s]+(?=[A-Z])'
    res_str = r'(?i:Results|Case\s+Report|Case|Clinical\s+Features)[:.\s]+(?=[A-Z])'
    con_str = r'(?i:Conclusions?)[:.\s]+(?=[A-Z])'
    comp_str = r'(?i:Competing Interest)'

    meth_match = re.search(meth_str, text)
    res_match = re.search(res_str, text)
    con_match = re.search(con_str, text)
    comp_match = re.search(comp_str, text)

    if con_match:
        con_end = comp_match.start() if comp_match else len(text)
        conclusion = text[con_match.end():con_end].strip()

    if res_match:
        res_end = con_match.start() if con_match else len(text)
        results = text[res_match.end():res_end].strip()

    if meth_match:
        meth_end = res_match.start() if res_match else (con_match.start() if con_match else len(text))
        method = text[meth_match.end():meth_end].strip()

    first_section = meth_match or res_match or con_match
    if first_section:
        introduction = text[:first_section.start()].strip()

    return {'aim': introduction, 'method': method, 'result': results, 'conclusion': conclusion}

def abstract2conclusion(text):
    marker = r'(?i:in conclusion,\s*)'
    match = re.search(marker, text)
    if match:
        return {'aim': None, 'method': None, 'result': text[:match.start()].strip(), 'conclusion': text[match.end():].strip()}
    return {'aim': None, 'method': None, 'result': text.strip(), 'conclusion': None}

def parse_case_report(text):
    """
    Extract case presentation and conclusion from a structured abstract.

    Logic:
    - conclusion: text after 'conclusion?:' until 'Competing Interest' or end
    - case presentation: text after 'case (presentation|report)?:' until conclusion or 'Competing Interest' or end
    - if no conclusion pattern, case presentation is everything after 'case presentation?:' until 'Competing Interest' or end
    - introduction - everything before the case pattern if no conclusion, otherwise everything before conclusion
    """
    case_pat = re.compile(r'case\s*(presentation|report)\s*:', re.IGNORECASE)
    con_pat = re.compile(r'conclusion\s*:', re.IGNORECASE)
    comp_pat = re.compile(r'Competing Interest', re.IGNORECASE)

    discussion = None
    case_presentation = None

    disc_match = con_pat.search(text)
    if disc_match:
        comp_match = comp_pat.search(text, disc_match.end())
        disc_end = comp_match.start() if comp_match else len(text)
        discussion = text[disc_match.end():disc_end].strip()
        body = text[:disc_match.start()]
    else:
        body = text

    case_match = case_pat.search(body)
    if case_match:
        if discussion:
            case_presentation = body[case_match.end():body.find(discussion)].strip()
        else:
            comp_match = comp_pat.search(body, case_match.end())
            case_end = comp_match.start() if comp_match else len(body)
            case_presentation = body[case_match.end():case_end].strip()

    return {'result': case_presentation, 'discussion': discussion}
def parse_abstract2(text):
    """
    Extract aim/introduction, methods, results, and conclusions from a structured abstract.

    Logic:
    - conclusions: text after 'conclusions?:' until 'Competing Interest' or end
    - methods and results (merged pattern): methods=None, results=text of merged section
    - standalone 'methods?:': methods=text until results, results=text until conclusions
    - no methods pattern at all: methods=None, results=None
    - aim: everything before the first matched section header
    """
    con_pat = re.compile(r'(?i:conclusions?)(?:\s+[A-Z]+)*\s*:')
    res_pat = re.compile(r'case\s+presentation\.|(?:results?|outcomes?|case(?:\s+(?:report|presentation|summary|description))?|clinical(?:\s+(?:report|presentation|features))?)\s*[:.]', re.IGNORECASE)
    meth_res_pat = re.compile(r'methods?\s+and\s+results?\s*:', re.IGNORECASE)
    meth_pat = re.compile(r'(?:materials\s+and\s+)?methods?\s*:', re.IGNORECASE)
    comp_pat = re.compile(r'Competing Interest', re.IGNORECASE)

    conclusions = None
    methods = None
    results = None
    aim = None

    # Extract conclusions
    con_match = con_pat.search(text)
    if con_match:
        comp_match = comp_pat.search(text, con_match.end())
        con_end = comp_match.start() if comp_match else len(text)
        conclusions = text[con_match.end():con_end].strip()
        body = text[:con_match.start()]
    else:
        body = text

    # Check for merged 'methods and results:' first
    meth_res_match = meth_res_pat.search(body)
    if meth_res_match:
        methods = None
        results = body[meth_res_match.end():].strip()
        aim = body[:meth_res_match.start()].strip() or None
    else:
        meth_match = meth_pat.search(body)
        if meth_match:
            res_match = res_pat.search(body, meth_match.end())
            if res_match:
                methods = body[meth_match.end():res_match.start()].strip()
                results = body[res_match.end():].strip()
            else:
                methods = body[meth_match.end():].strip()
                results = None
            aim = body[:meth_match.start()].strip() or None
        else:
            # No methods pattern — check for standalone result/case markers
            res_match = res_pat.search(body)
            if res_match:
                methods = None
                results = body[res_match.end():].strip()
                aim = body[:res_match.start()].strip() or None
            else:
                methods = None
                results = None
                aim = body.strip() or None

    return {'aim': aim, 'method': methods, 'result': results, 'conclusion': conclusions}


def bibtex_to_abstract(bib_path,output_path):
    with open(bib_path, 'r') as f:
        bib_data = parse_string(f.read(), bib_format='bibtex')
    output = []
    for entry in bib_data.entries.values():
        print(entry.fields['doi'])
        print(entry.fields['title'])
        output.append({
            'doi': entry.fields['doi'],
            'abstract': entry.fields['abstract']
        })
    with open(output_path, 'w') as f:
        json.dump(output, f)
def bibtex_to_conclusion_result(bib_path,output_path):
    """
    parse  data/bibtext.bib into entries,
    for each entry: 
    save 'doi', 'result', 'conclusion' to data/conclusion_result.json    
    """
    with open(bib_path, 'r') as f:
        bib_data = parse_string(f.read(), bib_format='bibtex')
    output = []
    for entry in bib_data.entries.values():
        print(entry.fields['doi'])
        print(entry.fields['title'])
        abstract = entry.fields['abstract']
        extracted_values=parse_abstract3(abstract)
        output.append({
            'doi': entry.fields['doi'],
            'conclusion': extracted_values.get('conclusion', ''),
            'result': extracted_values.get('result', ''),
            'abstract': abstract,
            
        })
    with open(output_path, 'w') as f:
        json.dump(output, f)
    #convert to dataframe and save as csv
    df = pd.DataFrame(output)
    df.to_csv(output_path.replace('.json', '.csv'), index=False)



def extract_bibtex_field(bib_string, field_name):
    """Extracts a specific field from a BibTeX string using brace counting."""
    pattern = re.compile(rf"{field_name}\s*=\s*", re.IGNORECASE)
    match = pattern.search(bib_string)
    
    if not match:
        return None
        
    start_pos = match.end()
    
    if start_pos < len(bib_string):
        first_char = bib_string[start_pos]
        
        if first_char == '{':
            brace_count = 1
            extracted = []
            for i in range(start_pos + 1, len(bib_string)):
                if bib_string[i] == '{':
                    brace_count += 1
                elif bib_string[i] == '}':
                    brace_count -= 1
                
                if brace_count == 0:
                    break
                extracted.append(bib_string[i])
            return "".join(extracted).strip()
            
        elif first_char == '"':
            end_pos = bib_string.find('"', start_pos + 1)
            if end_pos != -1:
                return bib_string[start_pos + 1:end_pos].strip()
                
        else:
            end_pos = bib_string.find(',', start_pos)
            if end_pos != -1:
                 return bib_string[start_pos:end_pos].strip()
                 
    return None

def parse_messy_bibtex(bib_string):
    """Returns a dictionary with only the requested fields."""
    return {
        "doi": extract_bibtex_field(bib_string, "doi"),
        "abstract": extract_bibtex_field(bib_string, "abstract"),
        "journal": extract_bibtex_field(bib_string, "journal")
    }

def process_bibtex_file(input_filepath)->pd.DataFrame:
    """Reads a file of messy BibTeX, extracts data, and return dataframe with doi, journal, conclusion and result."""
    results = []
    
    # 1. Open the file with utf-8 to safely handle non-ASCII characters
    with open(input_filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 2. Split the text into individual entries. 
    # (?m)^@ matches an '@' symbol ONLY at the start of a line.
    entries = re.split(r'(?m)^@', content)
    
    # 3. Process each entry
    for entry in entries:
        if not entry.strip():
            continue # Skip empty strings created by the split
            
        parsed_data = parse_messy_bibtex(entry)
        
        # Only keep it if we successfully found at least one of our target fields
        if parsed_data['doi'] or parsed_data['abstract']:
            #extract conclusion and result from abstract
            extracted_values=parse_abstract3(parsed_data['abstract'])
            parsed_data['conclusion']=extracted_values['conclusion']
            parsed_data['result']=extracted_values['result']
            results.append(parsed_data)    
    df = pd.DataFrame(results)
    df=df[['doi','journal','conclusion','result','abstract']]
    #convert sll emty strings to NaN
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    return df
def df2kaggle(df: pd.DataFrame, dir_out='out', name='biorxiv', subdir='result_conclusion',title=None,description=None):
    """Save df to <dir_out>/<subdir>/<name>/ and create dataset-metadata.json
    ready for 'kaggle datasets create -p <output_dir>'."""
    # 1. Create output directory structure
    output_dir = os.path.join(dir_out, subdir, name)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Save to CSV
    output_file = os.path.join(output_dir, f'{name}.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved dataset to {output_file}")

    # 3. Get Kaggle username from config
    kaggle_config_path = os.path.expanduser('~/.kaggle/kaggle.json')
    username = 'kaggle-user'
    if os.path.exists(kaggle_config_path):
        with open(kaggle_config_path, 'r') as f:
            kaggle_config = json.load(f)
            username = kaggle_config.get('username', 'kaggle-user')
    #replace any non-alphanumeric characters in name with hyphens for better compatibility with Kaggle dataset naming conventions
    name2 = re.sub(r'[^a-zA-Z0-9]+', '-', name).strip('-').lower()
    # 4. Create dataset-metadata.json file for Kaggle
    dataset_json = {
        "title": title,
        "id": f"{username}/{name2}-conclusions",
        "licenses": [{"name": "CC0"}],
        "resources": [{
            "path": f"{name}.csv",
            "description": description
        }]
    }
    json_file = os.path.join(output_dir, 'dataset-metadata.json')
    with open(json_file, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    print(f"Created dataset metadata at {json_file}")

    print(f"Ready to upload to Kaggle: kaggle datasets create -p {output_dir}")

def bib2kaggle(dir_data='data', dir_out='out', name='biorxiv', sample_size=50):
    """Reads a <dir_data>/<name>.bib of messy BibTeX, extracts data, samples and saves as csv,
    upload as public Kaggle dataset into directory result_conclusion/{name}."""
    # 1. Process the bibtex file
    input_file = os.path.join(dir_data, f'{name}.bib')
    print(f"Processing {input_file}...")
    df = process_bibtex_file(input_file)
    print(f"Loaded {len(df)} entries from BibTeX file")

    # 2. Drop if any column is missing or empty string, then sample
    df = df.dropna()
    print(f"{len(df)} entries after dropping missing values")

    if len(df) < sample_size:
        raise ValueError(f"{len(df)} entries in {input_file} after cleaning, not enough to sample {sample_size}")
    else:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {sample_size} entries")

    df2kaggle(df, dir_out=dir_out, name=name)
def fetch_pubmed_data_to_file(query, output_filename="pubmed_full_records.json"):
    # 1. Authenticate with NCBI
    Entrez.email = os.getenv("NCBI_EMAIL")
    #get frm environment variable or config file in production code
    Entrez.api_key =os.getenv("NCBI_API_KEY")  # Optional: Use an API key for higher rate limits
    print(f"Executing query: {query}")
    
    # 2. Search PubMed and cache the results on the History Server
    search_handle = Entrez.esearch(db="pubmed", term=query, usehistory="y")
    search_results = Entrez.read(search_handle)
    search_handle.close()
    
    count = int(search_results["Count"])
    print(f"Found {count} results matching your query.")
    
    if count == 0:
        print("No records to download. Exiting.")
        return

    # Extract the stateful server parameters required for downloading
    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]

    # 3. Fetch the actual records utilizing the Medline format
    # Note: For production scripts retrieving more than 10,000 records, 
    # you would need to wrap this in a loop using retstart and retmax.
    print("Downloading and parsing records...")
    fetch_handle = Entrez.efetch(
        db="pubmed",
        rettype="medline",
        retmode="text",
        retmax=count,
        webenv=webenv,
        query_key=query_key
    )
    
    # Parse the data stream iteratively using Bio.Medline
    records = Medline.parse(fetch_handle)
    #save the records to file for easier access later
    # Convert the iterator of dictionaries into a list
    records_list = list(records)

    # Save the list of dictionaries to a JSON file
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(records_list, f, indent=4)
    #To load this data back into a Python  use json.load(f)
    fetch_handle.close()
def records_to_df(records, date_cut="2025/10/01") -> pd.DataFrame:
    rows = []
    for record in records:
        pmid=record.get("PMID", "")
        phst=record.get("PHST", [])
        #skip record if any of phst values has date is between 1800 and 2025/10/01, phst values have format like '2025/05/27 00:00 [received]'
        if any([x for x in phst if x[:10] >= "1800/01/01" and x[:10] <= date_cut]):
            continue
        abstract = record.get("AB", "")

        doi = ""
        so = record.get("SO", "")
        if so:
            m = re.search(r'\bdoi:\s*(\S+)', so, re.IGNORECASE)
            if m:
                doi = m.group(1).rstrip('.')
        if not doi:
            lid = record.get("LID", [])
            aid = record.get("AID", [])
            lid = [lid] if isinstance(lid, str) else lid
            aid = [aid] if isinstance(aid, str) else aid
            for identifier in lid + aid:
                if "[doi]" in identifier:
                    parts = identifier.split()
                    idx = parts.index("[doi]")
                    if idx > 0:
                        doi = parts[idx - 1]
                    break

        mesh_list = record.get("MH", [])
        if isinstance(mesh_list, str):
            mesh_list = [mesh_list]
        mesh = "; ".join(mesh_list)

        pt_list = record.get("PT", [])
        if isinstance(pt_list, str):
            pt_list = [pt_list]
        pt = "; ".join(pt_list)

        parsed = parse_abstract2(abstract)
        if parsed['conclusion'] is None:
            parsed = parse_abstract3(abstract)
            if parsed['conclusion'] is None:
                parsed = abstract2conclusion(abstract)
        rows.append({
            "pmid": pmid,
            "doi": doi,
            "pt": pt,
            "mesh": mesh,
            "aim": parsed["aim"],
            "method": parsed["method"],
            "result": parsed["result"],
            "conclusion": parsed["conclusion"],
            "abstract": abstract,
        })
    df=pd.DataFrame(rows)
    #replace all empty strings with NaN
    df = df.replace(r'^\s*$', pd.NA, regex=True)
    return df

def records_to_csv(records, csv_file="pubmed_results.csv"):    
    # 4. Extract target data and write to CSV
    with open(csv_file, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Define the requested CSV headers
        writer.writerow(["abstract", "doi", "knowledge domain"])
        
        for record in records:
            # Get Abstract Text (AB key)
            abstract = record.get("AB", "")
            
            # Get DOI: try SO first, then LID, then AID
            doi = ""
            so = record.get("SO", "")
            if so:
                m = re.search(r'\bdoi:\s*(\S+)', so, re.IGNORECASE)
                if m:
                    doi = m.group(1).rstrip('.')
            if not doi:
                lid = record.get("LID", [])
                aid = record.get("AID", [])
                lid = [lid] if isinstance(lid, str) else lid
                aid = [aid] if isinstance(aid, str) else aid
                for identifier in lid + aid:
                    if "[doi]" in identifier:
                        parts = identifier.split()
                        idx = parts.index("[doi]")
                        if idx > 0:
                            doi = parts[idx - 1]
                        break
                    
            # Get Knowledge Domain (MeSH Terms are stored under the 'MH' key)
            # The API returns a list of strings, so we join them into a single readable string
            mesh_list = record.get("MH", [])
            if isinstance(mesh_list, str):
                mesh_list = [mesh_list]
            knowledge_domain = "; ".join(mesh_list)
            
            # Write the row to the CSV
            writer.writerow([abstract, doi, knowledge_domain])

# Execute the script with your exact query parameters
def pubmed2kaggle(name='pubmed', dir_data='data', dir_out='out', date_cut="2025/10/01",title=None,description=None):
    """Reads a <dir_data>/<name>.json  extracts data,sample and saves as csv,
    upload as public Kaggle dataset into directory result_conclusion/{name}.
    if PHST field(history of submission) present, it will skip record if any date is before the cut-off date
    """
    #check that dataset title must be between 6 and 50 characters
    if title and (len(title) < 6 or len(title) > 50):
        raise ValueError("Dataset title must be between 6 and 50 characters.")

    FILE_JSON = os.path.join(dir_data, f'{name}.json')
    with open(FILE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = records_to_df(data, date_cut=date_cut)
    df = df.dropna(subset=['doi', 'pmid','conclusion'])
    #check that it is unique by DOI
    if not df['doi'].is_unique:
        print("Warning: DOI is not unique, there are duplicates. see duplicates.csv for details.")
        duplicates = df[df.duplicated(subset=['doi'], keep=False)]
        duplicates.to_csv(os.path.join(dir_data, "duplicates.csv"), index=False)
        df = df.drop_duplicates(subset=['doi'], keep='first')
        #debugging statments:
        #df[df['conclusion'].isna()]['abstract'].apply(lambda x: print(f"Abstract with missing conclusion:\n{x}\n"))
    #count missing 'result'
    missing_result_count = df['result'].isna().sum()
    print(f"Number of entries:{df.shape[0]}, with missing 'result': {missing_result_count}. Even 'result' statment was not extracted, it might be merged into 'aim' or 'method' if the abstract is not well structured, so we keep those entries for now.")
    df2kaggle(df, dir_out=dir_out, name=name,title=title,description=description)
if __name__ == "__main__":
    abstract1="This study investigated the effects of curcumin (CUR)- and capsaicin (CAP)- loaded nanoemulsion on blood biochemical changes, oxidative status, jejunal morphology, inflammatory parameters, and performance of slow-growing Korat chickens (KRC) raised under high stocking density (HSD). A total of 480 male KRC (21 d of age) were allocated into four groups: (1) HSD without supplementation, (2) normal stocking density (NSD) without supplementation, (3) HSD supplemented with CUR and CAP in powdered form (P-CUR+CAP), and (4) HSD supplemented with CUR- and CAP-loaded nanoemulsions (NE-CUR+CAP). Chickens receiving NE-CUR+CAP showed no adverse changes in liver or kidney function compared with other groups. The heterophil-to-lymphocyte ratio was reduced in NE-CUR+CAP group relative to HSD group and was comparable with NSD and P-CUR+CAP groups (P < 0.05). NE-CUR+CAP also lowered levels of TBA in the liver and jejunum while enhancing hepatic superoxide dismutase activity compared with HSD group (P < 0.05). Villus height, villus height-to-crypt depth ratio, anti-inflammatory response, and cecal Lactobacillus counts were improved, whereas crypt depth and cecal Escherichia coli were reduced in the NE-CUR+CAP group (P < 0.05). Although feed intake, BW, and body weight gain were not affected, the feed conversion ratio was significantly lower in NE-CUR+CAP compared with HSD group (P < 0.05). In conclusion, NE-CUR+CAP mitigated oxidative stress and inflammation, improved intestinal health, and enhanced feed efficiency in slow-growing chickens raised under HSD."
    #for k,v in parse_abstract2(abstract1).items():print(f"{k}:\n{v}\n")
    #bib2kaggle(dir_data='data', name='biorxiv', sample_size=100)
    import dotenv
    dotenv.load_dotenv()
    USER_EMAIL = os.getenv("NCBI_EMAIL", "")
    # PUBMED_QUERY = '(conclusion*[Title/Abstract]) AND 2026/02/01:2026/02/02[crdt] NOT (1800:2025[epdat] OR 1800:2025[edat])';FILE_JSON='data/pubmed.json'
    PUBMED_QUERY = '(Case*[Publication Type]) AND (conclusion*[Title/Abstract]) AND 2026/02/01:2026/03/01[crdt] NOT (1800:2025[epdat] OR 1800:2025[edat])';    FILE_JSON='data/pubmed_case.json'
    #fetch_pubmed_data_to_file(query=PUBMED_QUERY,output_filename=FILE_JSON)
    pubmed2kaggle(name='pubmed_case', dir_data='data',description="Extracted case presentation and conclusion from PubMed abstracts with case reports, filtered by creation date and presence of 'conclusion' in title or abstract. Data includes PMID, DOI, publication type, MeSH terms, and parsed sections of the abstract.",title="PubMed Case Reports with Conclusions - pubmed_case")