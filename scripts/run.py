#!/usr/bin/env python
"""Load dataset, run LLM evaluation, and calculate scores."""

import json
import os
import re

import kaggle_benchmarks as kbench
import pandas as pd

from util.bench import (
    labels_completness,
    labels_grounding,
    make_context,
    _evaluate_completness,
    classify_grounding,
    calculate_score,
)


# --- Model configuration ---

model_slug = kbench.llm.model
if not model_slug:
    raise ValueError(
        "Model slug is not available. Please ensure that the LLM is properly "
        "configured and has a valid slug."
    )
print(f"Running evaluation for model: {model_slug}")


# --- Evaluation configuration ---

JUDGES = [
    # Order matters: grounding loop stops on first failure.
    # More strict judges come first.
    "openai/gpt-5.4-nano-2026-03-17",
    "anthropic/claude-opus-4-6@default",
    "google/gemini-3.1-pro-preview",
]
n_jobs = 10
n_examples = 100

file_score = "scores.json"
sample_score_max = (
    max(v for v in labels_completness.values() if v is not None)
    + max(v for v in labels_grounding.values() if v is not None)
)


# --- Data loading ---

current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

ds_name = "pubmed_case"
file_csv = f"{ds_name}.csv"
ds_name2 = re.sub(r"[^a-zA-Z0-9]+", "-", ds_name).strip("-").lower()
dataset_name = f"vmorozov/{ds_name2}-conclusions"

file_path = f"/kaggle/input/datasets/{dataset_name}/{file_csv}"
if "project" in current_dir:
    file_path = f"out/result_conclusion/{ds_name}/{file_csv}"  # local host
print(f"file path: {file_path}")

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    print("Datasets are not mounted. Fallback to use kagglehub to load the dataset.")
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_name, file_csv)


# --- Dataset filtering ---

if "pubmed" in ds_name:
    df = df[~df["pt"].str.contains("Retract", case=False, na=False)]
    df = df[~df["mesh"].isna()]
    df = df[~df["pt"].str.contains("Letter|Comment|Biography", case=False, na=False)]
    df = df[~df["result"].isna()]
if ds_name == "pubmed":
    df = df[~df["pt"].str.contains("Case", case=False, na=False)]


# --- Example preparation ---

if n_examples is not None:
    df = df.sample(n_examples, random_state=42)

examples = df.rename(columns={"conclusion": "reference_response", "doi": "id"})
examples["context_document"] = examples.apply(make_context, axis=1)
examples["full_prompt"] = examples.apply(
    lambda row: (
        "From the following research goals,methods and results generate a concise conclusion. "
        f"Do not use any outside knowledge or information.\n{row['context_document']}"
    ),
    axis=1,
)
examples["user_request"] = "Write a conclusion for research results."
columns = ["full_prompt", "user_request", "context_document", "reference_response", "id"]


# --- Task definition ---

@kbench.task(name="pubmed_case_conclusions")
def conclusion_grounded_task(
    llm, full_prompt, user_request, context_document, reference_response, id
) -> dict:
    try:
        response = llm.prompt(full_prompt).strip()
    except Exception as ex:
        print(f"Error getting llm response: {ex}")
        response = ""

    outputs = {"sample_id": id, "response": response}

    if not response:
        print(
            f"LLM response is empty for id {id}. Please ensure that the LLM is properly "
            f"configured and can generate a valid response. The provided prompt is: {full_prompt}"
        )
        outputs["quality"] = False
        outputs["grounding_score"] = 0
        return outputs

    quality_bool = []
    for judge in JUDGES:
        rating_bool, ans, rating = _evaluate_completness(
            user_request, response, reference_response, judge
        )
        quality_bool.append(rating_bool)
        outputs[f"quality-{judge}-text"] = ans
        outputs[f"quality-{judge}-rating"] = rating
        print(f"Judge {judge} says quality={rating}")

    outputs["quality"] = any(quality_bool)

    grounding_bools = []
    for judge in JUDGES:
        bool_ans, ans, parsed_answers = classify_grounding(
            user_request, context_document, response, judge
        )
        outputs[f"grounding-{judge}-text"] = ans
        outputs[f"grounding-{judge}-bool"] = bool_ans
        outputs[f"grounding-{judge}-parsed"] = parsed_answers
        grounding_bools.append(bool_ans)
        print(f"Judge {judge} says grounding={bool_ans}")

    outputs["grounding_score"] = sum(grounding_bools) / len(grounding_bools)
    return outputs


# --- Run evaluation ---

runs = conclusion_grounded_task.evaluate(
    llm=[kbench.llm],
    evaluation_data=examples[columns],
    n_jobs=3,
)


# --- Patch run files with model slug ---

run_files = kbench.kaggle.serialization.get_runs_filenames(runs)

for fp in run_files:
    with open(fp, "r") as f:
        run_data = json.load(f)
    if not run_data.get("modelVersion"):
        run_data["modelVersion"] = {"slug": model_slug}
    elif "slug" not in run_data["modelVersion"]:
        run_data["modelVersion"]["slug"] = model_slug
    with open(fp, "w") as f:
        json.dump(run_data, f)


# --- Local debugging: override run_files from an existing output directory ---

if "project" in os.getcwd():
    os.chdir("out/pubmed-clinical-case-conclusions/20260413_1800")
    run_files = [
        f
        for f in os.listdir()
        if f.endswith(".run.json") and not f.endswith("Run_aggregated.run.json")
    ]
    print(run_files)


# --- Aggregate and calculate scores ---

def _aggregate(run_results):
    return calculate_score(
        run_results,
        judges=JUDGES,
        file_score=file_score,
        completness_labels=labels_completness,
        grounding_labels=labels_grounding,
        sample_score_max=sample_score_max,
    )


kbench.kaggle.serialization.merge_results_from_runfiles(
    run_files=run_files,
    aggregate_fn=_aggregate,
    delete_run_files=False,
)
