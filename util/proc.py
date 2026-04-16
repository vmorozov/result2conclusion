import os
import json
import pandas as pd
from pathlib import Path
from pymongo import MongoClient

# --- Configuration ---
DB_NAME = "model_research"
COLLECTION_NAME = "runs"
MONGO_URI = "mongodb://localhost:27017/"  # Change if using Atlas
TARGET_KEY = "k11"  # The arbitrary key you are hunting for

def setup_mongo():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def ingest_json_files(collection, files_json):
    """ insert JSON files into MongoDB."""
    
    
    for file_path in files_json:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                # We add metadata so we know which file the data came from
                payload = {
                    "metadata": {
                        "path": str(file_path.parent),
                        "filename": file_path.name,
                        "folder": str(file_path.parent)
                    },
                    "data": data
                }
                collection.insert_one(payload)
            except json.JSONDecodeError:
                print(f"Error decoding {file_path}")
    
    print(f"Successfully ingested {len(files_json)} files.")

def find_key_recursive(obj, target, path=None):
    """Helper to crawl a document and yield paths + values."""
    if path is None: path = []
    
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = path + [k]
            if k == target:
                yield new_path, v
            if isinstance(v, (dict, list)):
                yield from find_key_recursive(v, target, new_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = path + [i]
            if isinstance(v, (dict, list)):
                yield from find_key_recursive(v, target, new_path)

def get_target_dataframe(collection, target_key):
    """Queries Mongo for docs containing the key and flattens them."""
    # Only pull documents where the key exists anywhere in the 'data' field
    # We use a wildcard index-style check or simply search the sub-object
    query = { f"data": { "$exists": True } } 
    cursor = collection.find(query)
    
    all_extracted_rows = []
    
    for doc in cursor:
        filename = doc['metadata']['filename']
        filepath = doc['metadata']['path']
        # Crawl the 'data' portion of the document
        for path, value in find_key_recursive(doc['data'], target_key):
            row = {
                "path": filepath,
                "file": filename,
                "value": value
            }
            # Dynamically add path levels (level_0, level_1, etc.)
            for i, p_val in enumerate(path):
                row[f"level_{i}"] = p_val
            
            all_extracted_rows.append(row)
            
    return pd.DataFrame(all_extracted_rows)

# --- Execution ---
if __name__ == "__main__":
    coll = setup_mongo()
    TARGET_KEY = "grounding-google/gemini-3.1-pro-preview-parsed"  # Change this to the key you want to hunt for
    # 1. Clear collection (optional) and Ingest
    coll.delete_many({})
    json_files = [path for path in Path("./").glob('**/*.run.json') if not path.name.endswith("Run_aggregated.run.json")] 
    print(f"Found {len(json_files)} JSON files to ingest.First 2 files: {[str(f) for f in json_files[:2]]}")
    ingest_json_files(coll, json_files)
    
    # 2. Query and Flatten
    df = get_target_dataframe(coll, TARGET_KEY)
    
    # 3. Clean up column order (File first, then Levels, then Value)
    cols = [c for c in df.columns if c not in ['path', 'file', 'value']]
    df = df[['path','file'] + sorted(cols) + ['value']]
    df.to_csv("extracted_values.csv", index=False)
    print(df.head(1).to_dict(orient='records')[0])  # Print first row as dict for clarity