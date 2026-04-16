import re
import json
from statistics import mean, stdev

import kaggle_benchmarks as kbench
import pandas as pd


# --- Label mappings ---

labels_grounding = {
    "not_supported": 0.0,
    "supported": 1.0,
    "no_rad": None,
    "unknown": None,
}

labels_completness = {
    "No Omissions": 1.0,
    "Minor Omission(s)": 0.5,
    "Major Omission(s)": 0.0,
    "Invalid": None,
    "unknown": None,
}


# --- Context helpers ---

def make_context(row):
    fields = [("aim", ""), ("method", "METHODS: "), ("result", "RESULTS: ")]
    return "\n".join(f"{prefix}{row[col]}" for col, prefix in fields if pd.notna(row[col])) + "\n"


# --- Completeness evaluation ---

def extract_instruction_json(ans: str) -> dict:
    pattern = re.compile(
        r'{\s*'
        r'"Instruction Following"\s*:\s*'
        r'"(No Omissions|Minor Omission\(s\)|Major Omission\(s\))"\s*'
        r'}'
    )
    match = pattern.search(ans)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception as e:
            print("!!Bad Json!!")
            print(e)
            print(ans)
            return {"Instruction Following": "Invalid"}
    return {"Instruction Following": "Invalid"}


def _evaluate_completness(user_request: str, response_a: str, response_b: str, judge: str):
    """Evaluates response_a for completeness using response_b as a reference."""
    prompt_template = """Your mission is to judge the response from an AI model, the *test* response for completeness using a *reference* response.
Please use the following rubric criteria to judge the responses:

<START OF RUBRICS>
Important: Only select `Major Omission(s)` for issues that are critical to the scientific results and conclusions. If the response misses minor details but still captures the essence of the scientific findings, it should be rated as `Minor Omission(s)`. If the response fully captures the scientific results and conclusions without missing any critical information, it should be rated as `No Omissions`.

Note: Both the `reference` and `test` responses were made based on a scientific result statement **which was redacted** in your analysis.
You do not have access to the result statement which might be referenced in the model responses (both reference and test) - assume that it contains the information referenced adequately.

In the end, express your final verdict as one of the following three json objects:

```json
{{
    "Instruction Following": "No Omissions"
}}
```

```json
{{
    "Instruction Following": "Minor Omission(s)"
}}
```

```json
{{
    "Instruction Following": "Major Omission(s)"
}}
```

<END OF RUBRICS>

# Your task
## User query
<|begin_of_query|>
{user_request}
<|end_of_query|>

**NOTE: CONTEXT DOCUMENT REDACTED FROM USER QUERY. ASSUME THAT IT CONTAINS THE NECESSARY INFORMATION WHICH IS REFERENCED IN THE RESPONSES BELOW.**

## Test Response:
<|begin_of_test_response|>
{response_a}
<|end_of_test_response|>

## Reference Response:
<|begin_of_reference_response|>
{response_b}
<|end_of_reference_response|>

Please write your analysis and final verdict for the test response.""".strip()

    prompt = prompt_template.format(
        user_request=user_request, response_a=response_a, response_b=response_b
    )
    system = "You are a helpful assistant. Your task is to evaluate the quality of a Response."
    llm_judge = kbench.llms[judge]
    try:
        with kbench.chats.new():
            evaluation_text = llm_judge.prompt(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            )
    except Exception as ex:
        print(f"Error getting llm response: {ex}")
        evaluation_text = ""

    parsed = extract_instruction_json(evaluation_text)
    return "Major Omission(s)" not in parsed["Instruction Following"], evaluation_text, parsed["Instruction Following"]


# --- Grounding evaluation ---

UFG_REV21_PROMPT_short = """
## Task: Evaluate Factual Accuracy of Model Response Against Provided Context

Your goal is to evaluate whether each sentence in the model's response is factually accurate **solely based on the provided context document**. You must act as a strict fact-checker. Do not use any external knowledge or make assumptions beyond what is stated in the text.

**Output Format:**
For each sentence in the response, output a JSON object with the following fields, representing an evaluation of a single sentence from the model's response:

* `sentence`: (string) The exact sentence from the model's response being evaluated.
* `label`: (string) Your verdict for the sentence. Must be one of:
    * `supported`: The sentence is fully and accurately supported by the provided context document.
    * `not_supported`: The sentence is NOT accurately supported by the provided context document. This includes sentences that are partially correct but contain inaccuracies, unsupported inferences, misattributions, or ignore crucial context.
    * `no_rad`: The sentence is introductory, concluding, conversational filler, or otherwise does not make a factual claim that can be verified against the context (e.g., "Here is the summary:", "Key aspects include:").
* `rationale`: (string) A detailed justification for your `label`.
    * If `supported`, explain *how* the provided excerpt(s) directly and accurately support the *entire* claim made in the sentence. Point out specific phrases or data points.
    * If `not_supported`, clearly explain *why* the sentence is inaccurate. Specify the inaccuracy (e.g., contradicts the text, makes an unsupported inference, misrepresents nuance, misattributes information, ignores contradictory context, uses incorrect terminology). Reference the relevant part(s) of the text or the lack thereof.
    * If `no_rad`, briefly explain why the sentence doesn't require factual verification against the context.
* `excerpt`: (string or null) One or more direct quotes (excerpts) from the context document that are **most relevant** to supporting or refuting the sentence. Concatenate multiple relevant excerpts with a semicolon and space (`; `). If the label is `not_supported` due to a complete lack of information, or if the label is `no_rad`, this field should be `null`.

**Evaluation Criteria - Strict Adherence Required:**

1.  **Explicit Support or Trivial/Strongly-Implied Inferences Only:** A sentence is only `supported` if the information is **explicitly stated** in the context or are logically inferred statements that are strongly implied by the context or trivially implied. Do NOT mark sentences as `supported` if they require inference, assumption, synthesis beyond direct statements, or understanding of implied meaning beyond a very strong or trivial implication. A non-trivial inference that goes well beyond what is strongly or trivially implied by the context is NOT supported.
2.  **Precision and Nuance:** Pay close attention to specific terminology, qualifiers (e.g., "significantly", "most", "some"), quantities, and causal relationships mentioned in the text. Simplifications in the model response are only acceptable if they maintain **full factual accuracy** and do not misrepresent the nuance of the source.
3.  **Completeness and Context:** Evaluate the sentence in the context of the *entire* provided document. A sentence is `not_supported` if it cherry-picks information while ignoring contradictory or significantly qualifying information present elsewhere *in the same document*. The response should not be misleading even if technically based on a fragment of the text.
4.  **Accurate Attribution:** Ensure that claims, descriptions, or characteristics are attributed to the correct subject, object, or scope as defined in the text. Misattributions make a sentence `not_supported`.
5.  **No Hallucinations:** The model response must not introduce information, terms, or concepts not present in the context. Such additions make a sentence `not_supported`.
6.  **Sentence-Level Evaluation:** Evaluate each sentence independently based on these criteria.

Output each JSON object on a new line.

**Example Structure:**

```json
{{"sentence": "The first sentence of the model response.", "label": "supported/not_supported/no_rad", "rationale": "Detailed explanation referencing the criteria and text.", "excerpt": "Relevant quote(s) from context; null if none."}}
{{"sentence": "The second sentence of the model response.", "label": "supported/not_supported/no_rad", "rationale": "Detailed explanation referencing the criteria and text.", "excerpt": "Relevant quote(s) from context; null if none."}}
// ... more sentence evaluations
```

**DO NOT** use RAW quotation marks (") in the values of the sentence, rationale, or excerpt fields in the response. Replace any quotations (") with tick (`)
Example: "The response states, `This is the way it goes`". DO NOT attempt any escaping of the quotations marks - just replace them with ticks.

**Now, please analyze the following context and response:**

**User Query:**
{user_query}

**Context:**
{context}

**Response:**
{response}
""".strip()


def clean_json_content_8(text: str, start_field: str, end_field: str | None = None) -> str:
    r"""Cleans the value of a specified field in a JSON-like string."""
    if end_field:
        trailer_pattern = rf'(,\s*"{re.escape(end_field)}":)'
    else:
        trailer_pattern = r"(\s*})"

    pattern = re.compile(
        rf'"{re.escape(start_field)}":(.+?){trailer_pattern}', re.DOTALL
    )

    def _clean_match(match: re.Match[str]) -> str:
        content = match.group(1)
        trailer = match.group(2)
        content = content.replace('"', "")
        content = content.replace("\\", "")
        content = content.replace("\x02", "")
        content = content.strip()
        cleaned_content = f'"{content}"' if content != "null" else "null"
        return f'"{start_field}":{cleaned_content}{trailer}'

    cleaned_text, num_subs = pattern.subn(_clean_match, text, count=1)
    return cleaned_text if num_subs > 0 else text


def parse_ufg_rev21_verdict(
    answer: str,
    judge: str,
    verbose: bool = False,
    labels_to_return=labels_grounding.keys(),
) -> tuple[bool, list[dict[str, str]]]:
    """Parses a structured JSON answer into a boolean grounding prediction."""
    if "```json" in answer:
        jsonl_chunks = []
        for a in answer.split("```json")[1:]:
            jsonl_chunks.append(a.split("```")[0])
        answer = "\n".join(jsonl_chunks)
    answer = answer.strip()
    answer = answer.replace("}\n", "}\n@\n@\n")
    answer = answer.replace("} \n", "}\n@\n@\n")
    answer = answer.replace("}  \n", "}\n@\n@\n")
    answer = answer.replace("<ctrl75>", "")

    parsed_answers = []
    for line in answer.split("\n@\n@\n"):
        raw_line = line
        try:
            line = line.replace("\n", " ")
            line = line.replace("\\'", "'")
            line = line.replace("<ctrl75>", "")
            line = line.lstrip(",")
            line = line.replace("\n", " ")
            line = clean_json_content_8(line, "sentence", "label")
            line = clean_json_content_8(line, "label", "rationale")
            line = clean_json_content_8(line, "rationale", "excerpt")
            line = clean_json_content_8(line, "excerpt")
            parsed = json.loads(line)
            if "label" not in parsed:
                parsed["label"] = "unknown"
            parsed_answers.append(parsed)
        except (json.JSONDecodeError, ValueError, TypeError):
            if verbose:
                print(
                    f"error with parsing sentence from {judge} model response to json. "
                    f"The full raw line is: {raw_line}"
                )
            if '"not_supported"' in raw_line:
                parsed_answers.append(
                    {"sentence": "", "rationale": "", "excerpt": "", "label": "not_supported"}
                )

    if parsed_answers:
        bool_ans = all(
            p["label"] in ("supported", "no_rad", "unknown") for p in parsed_answers
        )
    else:
        if verbose:
            print(
                f"error with parsing json response from {judge} model. "
                f"The full answer is: {answer}"
            )
        bool_ans = False

    filtered = [p for p in parsed_answers if p["label"] in labels_to_return]
    return bool_ans, filtered


def classify_grounding(user_query: str, context: str, response: str, judge: str):
    prompt = UFG_REV21_PROMPT_short.format(
        user_query=user_query, context=context, response=response
    )
    system = (
        "You are a helpful assistant. You will be provided with a textual context and a "
        "model-generated response. Your task is to analyze the response sentence by sentence "
        "and classify each sentence according to its relationship with the provided context."
    )
    llm_judge = kbench.llms[judge]
    try:
        with kbench.chats.new():
            ans = llm_judge.prompt(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            )
    except Exception as ex:
        print(f"Error get grader llm response: {ex}")
        ans = ""
    bool_ans, parsed_answers = parse_ufg_rev21_verdict(ans, judge, verbose=False)
    return bool_ans, ans, parsed_answers


# --- Scoring ---


def count_labels(response_string: str, target_labels: set[str]) -> dict[str, int]:
    """Counts occurrences of specified labels in a response string without converting to dictionary. Just match "<label>" strings."""
    count = {label: 0 for label in target_labels}
    for line in response_string.splitlines():
        line = line.strip()
        for label in target_labels:
            if f'"{label}"' in line:
                count[label] += 1
    return count
def extract_labels_from_response(response_string: str) -> list[str]:
    labels = []
    for line in response_string.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                labels.append(data.get("label", "unknown"))
            except json.JSONDecodeError:
                continue
    return labels


def calculate_accuracy_and_ci(run_results: list[dict]) -> float:
    scores = []
    for result in run_results:
        if "dictResult" in result:
            score = result["dictResult"].get("grounding_score", None)
            scores.append(score)
    df = pd.DataFrame({"grounding_score": scores})
    return float(df.grounding_score.mean()) if len(df) > 0 else 0.0


def calculate_score(
    run_results: list[dict],
    judges: list[str],
    file_score: str,
    completness_labels: dict,
    grounding_labels: dict,
    sample_score_max: float,
) -> tuple[float, float]:
    """Compute mean and stdev of normalised per-sample scores and persist them to file_score."""
    scoring = {}

    for result in run_results:
        if "dictResult" not in result:
            continue
        d = result["dictResult"]
        if "sample_id" not in d:
            continue
        sample_id = d["sample_id"]
        response = d.get("response") or ""
        if len(response.strip())==0:
            continue
        scoring[sample_id] = {"response_length": len(response), "quality": {}, "grounding": {}}

        for judge in judges:
            rating_key = f"quality-{judge}-rating"
            if rating_key in d:
                scoring[sample_id]["quality"][judge] = completness_labels.get(d[rating_key], None)

            grounding_key = f"grounding-{judge}-text"
            if grounding_key in d:
                grounding_text = d[grounding_key]
                parsed_labels = extract_labels_from_response(grounding_text)
                
                if len(parsed_labels) == 0 or not all(
                    grounding_labels[x] for x in parsed_labels if x in grounding_labels
                ):
                    # print(
                    #     f"no labels with simple parser for grounding evaluation "
                    #     f"for sample_id {sample_id} {grounding_key}"
                    # )
                    
                    _, parsed = parse_ufg_rev21_verdict(grounding_text, judge, verbose=False)
                    parsed_labels = [p["label"] for p in parsed]
                label_count=count_labels(grounding_text, set(grounding_labels.keys()))
                #count parsed_labels
                c2={label: parsed_labels.count(label) for label in grounding_labels.keys()}
                if label_count != c2:
                    print(
                        f"label counting mismatch for sample_id {sample_id} {grounding_key} "
                        f"simple counting: {label_count} vs json parsing: {c2}"
                    )
                if sum(label_count.values()) > 0:
                    #subset label_count for labels that has None value in grounding_labels
                    label_count2 = {label: count for label, count in label_count.items() if grounding_labels[label] is not None}
                    #calculate mean based on count of labels
                    score = sum(
                        count * grounding_labels[label]
                        for label, count in label_count2.items()
                        if label in grounding_labels and grounding_labels[label] is not None
                    ) / sum(label_count2.values())
                    scoring[sample_id]["grounding"][judge] = {"score": score, "label_count": label_count}
                else:
                    print(
                        f"no labels were extracted for grounding evaluation "
                        f"for sample_id:{sample_id} {grounding_key}:{grounding_text}"
                    )
                    scoring[sample_id]["grounding"][judge] = None
    #print current directory and check if file_score already exists
    #
    df = score2df(scoring)
    df.to_csv(file_score.replace(".json", ".csv"), index=False)
    with open(file_score, "w") as f:
        json.dump(scoring, f, indent=4)

    scores = []
    for sample_data in scoring.values():
        quality_vals = [v for v in sample_data["quality"].values() if v is not None]
        grounding_vals = [
            v["score"] for v in sample_data["grounding"].values()
            if v is not None
        ]
        sample_score = sum(
            mean(vals)
            for vals in (quality_vals, grounding_vals)
            if vals
        )
        scores.append(sample_score / sample_score_max)
    valid = [s for s in scores if s is not None]
    if len(valid) == 0:
        return 0.0, 0.0
    return mean(valid), stdev(valid)
def score2df(scoring: dict) -> pd.DataFrame:
    """Convert the record scoring dictionary like: "10.1186/s12879-026-12947-x": {
        "response_length": 331,
        "quality": {
            "google/gemma-4-31b": 0.5,
            "qwen/qwen3-235b-a22b-instruct-2507": 0.0
        },
        "grounding": {
            "google/gemma-4-31b": {
                "score": 1.0,
                "label_count": {
                    "supported": 2,
                    "not_supported": 0,
                    "unknown": 0,
                    "no_rad": 0
                }
            },
            "qwen/qwen3-235b-a22b-instruct-2507": {
                "score": 1.0,
                "label_count": {
                    "supported": 2,
                    "not_supported": 0,
                    "unknown": 0,
                    "no_rad": 0
                }
            }
        }
    } to a pandas DataFrame with columns: sample_id, response_length, metric(e.g. "quality"),judge, score,label_count(sum of "label_count" for grounding or None for quality)"""
    rows = []
    columns = ["sample_id", "response_length", "metric", "judge", "score", "label_count"]

    for sample_id, sample_data in scoring.items():
        response_length = sample_data.get("response_length")

        for judge, score in (sample_data.get("quality") or {}).items():
            rows.append(
                {
                    "sample_id": sample_id,
                    "response_length": response_length,
                    "metric": "quality",
                    "judge": judge,
                    "score": score,
                    "label_count": None,
                }
            )

        for judge, grounding_data in (sample_data.get("grounding") or {}).items():
            if grounding_data is None:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "response_length": response_length,
                        "metric": "grounding",
                        "judge": judge,
                        "score": None,
                        "label_count": None,
                    }
                )
                continue

            counts = grounding_data.get("label_count") or {}
            rows.append(
                {
                    "sample_id": sample_id,
                    "response_length": response_length,
                    "metric": "grounding",
                    "judge": judge,
                    "score": grounding_data.get("score"),
                    "label_count": sum(counts.values()) if counts else 0,
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(["sample_id", "metric", "judge"], na_position="last").reset_index(drop=True)
    
if __name__ == "__main__":
    # Example usage:
    with open("scores.json", "r") as f:
        scoring_data = json.load(f)
    df = score2df(scoring_data)
    df.to_csv("scores.csv", index=False)
    print(df.head())
