import json
import os
import re
from pathlib import Path

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
JUDGES = [
    # Order matters: grounding loop stops on first failure.
    # More strict judges come first.
    "openai/gpt-5.4-nano-2026-03-17",
    "anthropic/claude-opus-4-6@default",
    "google/gemini-3.1-pro-preview",
]
JUDGES=['anthropic/claude-haiku-4-5@20251001', 'anthropic/claude-opus-4-1@20250805', 'anthropic/claude-opus-4-5@20251101', 'anthropic/claude-opus-4-6@default', 'anthropic/claude-sonnet-4-5@20250929', 'anthropic/claude-sonnet-4-6@default', 'anthropic/claude-sonnet-4@20250514', 'deepseek-ai/deepseek-r1-0528', 'deepseek-ai/deepseek-v3.1', 'deepseek-ai/deepseek-v3.2', 'google/gemini-2.0-flash', 'google/gemini-2.0-flash-lite', 'google/gemini-2.5-flash', 'google/gemini-2.5-pro', 'google/gemini-3-flash-preview', 'google/gemini-3.1-flash-lite-preview', 'google/gemini-3.1-pro-preview', 'google/gemma-3-12b', 'google/gemma-3-1b', 'google/gemma-3-27b', 'google/gemma-3-4b', 'google/gemma-4-26b-a4b', 'google/gemma-4-31b', 'openai/gpt-5.4-2026-03-05', 'openai/gpt-5.4-mini-2026-03-17', 'openai/gpt-5.4-nano-2026-03-17', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3-235b-a22b-instruct-2507', 'qwen/qwen3-coder-480b-a35b-instruct', 'qwen/qwen3-next-80b-a3b-instruct', 'qwen/qwen3-next-80b-a3b-thinking', 'zai/glm-5']

file_score = "scores.json"
sample_score_max = (
    max(v for v in labels_completness.values() if v is not None)
    + max(v for v in labels_grounding.values() if v is not None)
)


# --- Data loading ---
def _aggregate(run_results):
    return calculate_score(
        run_results,
        judges=JUDGES,
        file_score=str(run_dir / "scores.json"),
        completness_labels=labels_completness,
        grounding_labels=labels_grounding,
        sample_score_max=sample_score_max,
    )

current_dir = Path.cwd()
print(f"Current directory: {current_dir}")


def _iter_run_directories(out_dir: Path):
    excluded_dirs = [out_dir / "result_conclusion",out_dir / "conclusion-consistent"]

    for root, dirs, files in os.walk(out_dir):
        root_path = Path(root)
        #process only with '2026041' in path
        if "2026040"  in str(root_path): #skip old results
            continue
        # Do not traverse the excluded subtree.
        dirs[:] = [
            d for d in dirs if not any((root_path / d).resolve().is_relative_to(excluded_dir.resolve()) for excluded_dir in excluded_dirs)
        ]

        run_files = sorted(
            str(root_path / file_name)
            for file_name in files
            if file_name.endswith(".run.json")
            and not file_name.endswith("Run_aggregated.run.json")
        )
        if run_files:
            yield root_path, run_files


if "project" in str(current_dir):
    out_dir = current_dir / "out"

    for run_dir, run_files in _iter_run_directories(out_dir):
        print(f"Processing directory: {run_dir}")
        print(f"run_files[:3]: {run_files[:3]}")

        kbench.kaggle.serialization.merge_results_from_runfiles(
            run_files=run_files,
            aggregate_fn=_aggregate,
            delete_run_files=False,
            output_directory=str(run_dir),
        )
