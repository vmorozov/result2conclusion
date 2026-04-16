#!/usr/bin/env python
# coding: utf-8

# ## Benchmark designed to evaluate the completeness and factual accuracy of AI generated conclusions from clinical case background and presentation. 

# ### Example of scientific abstract:
# BACKGROUND Solitary fibrous tumors (SFTs) are rare mesenchymal neoplasms most commonly originating in the pleura but may also arise in extrapleural sites, including the head and neck region. Subcutaneous SFTs in the chin area are exceptionally rare; only a few cases have been reported. CD34 and Bcl-2 are commonly used immunohistochemical markers in the initial diagnostic workup, and their co-expression is strongly suggestive of an SFT. However, CD34 negativity can be misleading and complicate diagnosis. Accurate identification of SFTs is potentially difficult because of histologic variability and overlap with other spindle cell tumors. CASE REPORT We encountered a rare subcutaneous SFT in the chin of a 22-year-old woman who presented with facial asymmetry and a tender mass. Imaging revealed a well-circumscribed, contrast-enhancing lesion in subcutaneous soft tissue over the anterior mandible. Histologically, the tumor lacked typical staghorn vasculature but showed spindle cell proliferation within a fibrous stroma. Immunohistochemistry demonstrated strong nuclear STAT6 positivity; expression of CD99, Bcl-2, and SMA; and focal H-caldesmon staining. CD34, S100, and ALK1 displayed negative staining. Based on these findings, the patient was diagnosed with an SFT. CONCLUSIONS This case highlights the importance of considering SFT in the differential diagnosis of spindle cell tumors in the head and neck region. It also underscores the critical role of immunohistochemistry, particularly STAT6 staining, in distinguishing SFT from histologic mimics. Vigilant follow-up remains essential, especially in atypical or CD34-negative cases, given their potential for aggressive behavior.
# 

# Conclusions and required context were extracte from PubMed abstracts with https://github.com/vmorozov/result2conclusion

# This notebook will :
# 
# 1) **Generate conclusion** from  context sections
# 
# 2) **Evaluate response** with multiple LLMs("judges") using the real conclusion section from the abstract as reference
# 
# 3) **Ensemble evaluations** to produce a score
# 

# The LMM evaluation consists of following steps: 
#     1) "completeness" where LLM judges check for omissions comparing to the reference conclusion statement 
#     2) "grounding" where LLM judges check if generated sencences are supported by the result statement
#     3) the sample score is sum of "completeness" and "grounding" scores.  Multiple judges are averaged
# 
# The notebook structure is borrowed from the [FACTS Grounding benchmark](https://www.kaggle.com/code/prathameshbang/facts-grounding-v2-benchmark-starter). 

# In[1]:


import kaggle_benchmarks as kbench

import re
import json


# In[2]:


#see kbench.llms avalible models
#if hasattr(kbench, 'llms') and isinstance(kbench.llms, dict):    print(list(kbench.llms.keys()))
## ['anthropic/claude-haiku-4-5@20251001', 'anthropic/claude-opus-4-1@20250805', 'anthropic/claude-opus-4-5@20251101', 'anthropic/claude-opus-4-6@default', 'anthropic/claude-sonnet-4-5@20250929', 'anthropic/claude-sonnet-4-6@default', 'anthropic/claude-sonnet-4@20250514', 'deepseek-ai/deepseek-r1-0528', 'deepseek-ai/deepseek-v3.1', 'deepseek-ai/deepseek-v3.2', 'google/gemini-2.0-flash', 'google/gemini-2.0-flash-lite', 'google/gemini-2.5-flash', 'google/gemini-2.5-pro', 'google/gemini-3-flash-preview', 'google/gemini-3.1-flash-lite-preview', 'google/gemini-3.1-pro-preview', 'google/gemma-3-12b', 'google/gemma-3-1b', 'google/gemma-3-27b', 'google/gemma-3-4b', 'google/gemma-4-26b-a4b', 'google/gemma-4-31b', 'openai/gpt-5.4-2026-03-05', 'openai/gpt-5.4-mini-2026-03-17', 'openai/gpt-5.4-nano-2026-03-17', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'qwen/qwen3-235b-a22b-instruct-2507', 'qwen/qwen3-coder-480b-a35b-instruct', 'qwen/qwen3-next-80b-a3b-instruct', 'qwen/qwen3-next-80b-a3b-thinking', 'zai/glm-5']


# In[3]:


model_slug = kbench.llm.model
#raise error if model_slug is None or empty
if not model_slug:
    raise ValueError("Model slug is not available. Please ensure that the LLM is properly configured and has a valid slug.")

print(f"Running evaluation for model: {model_slug}")


# In[4]:


JUDGES = [
    #order is important here,becuse grounding evaluation loop stop if judge calls failure. We assume that some judges may be more strict than others and we want to make sure that if a response fails a stricter judge, it is not evaluated by a more lenient judge which may give it a passing score
    #'google/gemini-3.1-flash-lite-preview',
    # 'google/gemini-2.5-flash',
    'openai/gpt-5.4-nano-2026-03-17',
    "anthropic/claude-opus-4-6@default",
    "google/gemini-3.1-pro-preview"

]
n_jobs=10
n_examples=100

labels_grounding = {"not_supported":0.0,"supported": 1.0, "no_rad": None, "unknown": None}
labels_completness = {
    "No Omissions": 1.0,
    "Minor Omission(s)": 0.5,
    "Major Omission(s)": 0.0,
    "Invalid": 0.0,
    "unknown": None
}
file_score = f"scores.json"
sample_score_max=max(filter(lambda x: x is not None, labels_completness.values()))+max(filter(lambda x: x is not None, labels_grounding.values()))


# In[5]:


import os
import pandas as pd
import re
#get current directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")
ds_name="pubmed_case"
file_csv = f"{ds_name}.csv"
ds_name2=re.sub(r'[^a-zA-Z0-9]+', '-', ds_name).strip('-').lower()
dataset_name = f"vmorozov/{ds_name2}-conclusions"
file_path = f"/kaggle/input/datasets/{dataset_name}/{file_csv}" 
if 'project' in current_dir:file_path = f"out/result_conclusion/{ds_name}/{file_csv}" #local host
print(f"file path:{file_path}")
#supposed to be: /kaggle/input/datasets/vmorozov/pubmed-case-conclusions/pubmed_case.csv
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    print("Datasets are not mounted. Fallback to use kagglehub to load the dataset.")
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
        dataset_name,
        file_csv
    )


# In[6]:


if "pubmed" in ds_name:
    df=df[~df['pt'].str.contains('Retract', case=False, na=False)] #exclude retracted papers
    df=df[~df['mesh'].isna()] #exclude records with empty "mesh" field, mesh field might be useful for post-processing analysis of results, but it is not used for prompting and we want to make sure that all records have it filled to avoid biasing the sample with records that have empty mesh field. Also, empty mesh field might be a sign of lower quality record.
    df=df[~df['pt'].str.contains('Letter|Comment|Biography', case=False, na=False)] #exclude some less quality publication types
    df=df[~df['result'].isna()] #if 'result' (case presentation/description) wasn't extracted, it might be unusual abstract structure
if ds_name=="pubmed":
    df=df[~df['pt'].str.contains('Case', case=False, na=False)]#exclude clinical case reports for the separate clinical benchmark
    #exclude with empty "mesh"



# In[7]:


def make_context(row):
    fields = [('aim', ''), ('method', 'METHODS: '), ('result', 'RESULTS: ')]
    return "\n".join(f"{prefix}{row[col]}" for col, prefix in fields if pd.notna(row[col])) + "\n"



# In[8]:


if n_examples is not None:
    df=df.sample(n_examples, random_state=42) 
examples=df.rename(columns={'conclusion':'reference_response', 'doi':'id'})
examples['context_document'] = examples.apply(make_context, axis=1)
examples['full_prompt'] = examples.apply(lambda row: f"From the following research goals,methods and results generate a concise conclusion. Do not use any outside knowledge or information.\n{row['context_document']}", axis=1)
examples['user_request'] = f"Write a conclusion for research results."
columns = ['full_prompt','user_request','context_document','reference_response','id']
examples.head(5)['context_document'].to_list()


# ## Quality Prompts and Helper Functions

# In[9]:


def extract_instruction_json(ans: str) -> dict:
    pattern = re.compile(
        r'{\s*'
        r'"Instruction Following"\s*:\s*'
        r'"(No Omissions|Minor Omission\(s\)|Major Omission\(s\))"\s*'
        r'}'
    )

    match = pattern.search(ans)

    if match:
        json_string = match.group(0)
        try:
            # Once the valid JSON string is found, parse it
            return json.loads(json_string)
        except Exception as e:
            # This is unlikely to happen if the regex matches, but it's good practice
            print("!!Bad Json!!")
            print(e)
            print(ans)
            return {"Instruction Following": "Invalid"}

    return {"Instruction Following": "Invalid"}


#Completness rating rubric:

def _evaluate_completness(
    user_request: str, response_a: str, response_b: str, judge: str
):
    """Evaluates response_a for completness using response_b as a reference."""
    ineligible_responses_filter_no_context_prompt = """Your mission is to judge the response from an AI model, the *test* response for completeness using a *reference* response.
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

    prompt = ineligible_responses_filter_no_context_prompt.format(
        user_request=user_request, response_a=response_a, response_b=response_b
    )
    s = "You are a helpful assistant. Your task is to evaluate the quality of a Response."
    judge = kbench.llms[judge]
    try:
        with kbench.chats.new():
            evaluation_text = judge.prompt([{"role": "system", "content": s}, {"role": "user", "content": prompt}])
    except Exception as ex:
        print(f"Error getting llm response: {ex}")
        evaluation_text = ""

    parsed = extract_instruction_json(evaluation_text)

    return 'Major Omission(s)' not in parsed['Instruction Following'], evaluation_text, parsed['Instruction Following']


# ## Grounding Prompts and Helper Functions

# In[10]:


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

def clean_json_content_8(
    text: str, start_field: str, end_field: str | None = None
) -> str:
  r"""Cleans the value of a specified field in a JSON-like string.

  This function can clean a field in the middle of a JSON object or the
  very last field. The output of a model might be a string like:

  {
    "sentence": "The first sentence.",
    "rationale": "The first rationale.",
    "excerpt": "  \\"The first excerpt.\\"  ",
    "label": "supported"
  }

  This function can target a specific field's value (e.g., "excerpt"),
  remove extra quotes and escape characters, and reconstruct the string.

  Args:
    text: The full JSON-like string to clean.
    start_field: The name of the field whose value needs to be cleaned.
    end_field: The name of the field that immediately follows the start_field.
      If start_field is the last field in the object, this should be set to
      None.

  Returns:
    The cleaned string if the specified field is found, otherwise the
    original string.
  """

  # The part of the regex that looks for what comes *after* the value.
  # It's either the next field or the closing brace of the JSON object.
  if end_field:
    # Pattern for a field followed by another field.
    # Captures the comma and the next field key to preserve it.
    trailer_pattern = rf'(,\s*"{re.escape(end_field)}":)'
  else:
    # Pattern for the last field in the object.
    # Captures optional whitespace and the closing brace to preserve it.
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
    answer: str, judge: str, verbose: bool = False
,labels_to_return=labels_grounding.keys()) -> tuple[bool, list[dict[str, str]], bool]:
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
      # pylint-enable: g-inconsistent-quotes
      line = line.lstrip(",")
      #  Remove any additional newlines that manifest
      #  as a result of the above replacements.
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
            f"error with parsing sentence from {judge} model response to json. The full"
            f" raw line is: {raw_line}"
        )

      if '"not_supported"' in raw_line:
        parsed_answers.append({
            "sentence": "",
            "rationale": "",
            "excerpt": "",
            "label": "not_supported",
        })
  if parsed_answers:
    bool_ans = all(
        parsed_answer["label"] in ("supported", "no_rad", "unknown")
        for parsed_answer in parsed_answers
    )
  else:
    if verbose:
      print(
          f"error with parsing json response from {judge} model. The full answer is:"
          f" {answer}"
      )
    bool_ans = False
  # Only returns unsupported and contradictory sentences to use for rationales.
  filtered_parsed_answers = [
      parsed_answer
      for parsed_answer in parsed_answers
      if parsed_answer["label"] in labels_to_return
  ]

  return bool_ans, filtered_parsed_answers

def classify_grounding(user_query, context, response, judge):
    prompt = UFG_REV21_PROMPT_short.format(user_query=user_query, context=context, response=response)
    s = "You are a helpful assistant. You will be provided with a textual context and a model-generated response. Your task is to analyze the response sentence by sentence and classify each sentence according to its relationship with the provided context."
    judge = kbench.llms[judge]
    try:
        with kbench.chats.new():
            ans = judge.prompt([
                {"role": "system", "content": s},
                {"role": "user", "content": prompt}],)
    except Exception as ex:
        print(f"Error get grader llm response: {ex}")
        ans =""
    bool_ans, parsed_answers = parse_ufg_rev21_verdict(ans, judge, verbose=False)
    return bool_ans, ans, parsed_answers


# In[11]:


@kbench.task(name="pubmed_case_conclusions")
def conclusion_grounded_task(llm,full_prompt,user_request,context_document,reference_response,id)-> dict:
    try:
        response = llm.prompt(full_prompt).strip()
    except Exception as ex:
        print(f"Error getting llm response: {ex}")
        response=""    
    outputs = {}
    outputs["sample_id"] = id
    outputs["response"] = response
    #if response is empty, return False for quality and 0 for grounding score, and skip judges evaluation since there is no response to evaluate
    if not response:
        #throw warning instead of error since we want to keep track of these cases in the output and not fail the entire evaluation run  
        print(f"LLM response is empty for id {id}. Please ensure that the LLM is properly configured and can generate a valid response.The provided prompt is: {full_prompt}")
        outputs["quality"] = False
        outputs["grounding_score"] = 0
        return outputs

    quality_bool = []
    for judge in JUDGES:
        rating_bool, ans, rating = _evaluate_completness(
            user_request,
            response,
            reference_response,
            judge,
        )
        quality_bool.append(rating_bool)
        outputs[f"quality-{judge}-text"] = ans
        outputs[f"quality-{judge}-rating"] = rating
        print(f"Judge {judge} says quality={rating}")

        # If one judge says the quality is good, we break out of the loop since future calls to judges for the same response are not needed.
        # if rating_bool:
        #     quality_bool = True
        #     break
    outputs["quality"] = any(quality_bool)

    grounding_bools = []
    for judge in JUDGES:
        bool_ans, ans, parsed_answers = classify_grounding(user_request, context_document, response, judge)
        outputs[f"grounding-{judge}-text"] = ans
        outputs[f"grounding-{judge}-bool"] = bool_ans
        outputs[f"grounding-{judge}-parsed"] = parsed_answers
        grounding_bools.append(bool_ans)
        print(f"Judge {judge} says grounding={bool_ans}")
    outputs["grounding_score"] = sum(grounding_bools) / len(grounding_bools)

    return outputs


# In[12]:


# Evaluate the task on the dataset
runs = conclusion_grounded_task.evaluate(
    llm=[kbench.llm],
    evaluation_data=examples[columns],
    n_jobs=3,
)


# In[13]:


def calculate_accuracy_and_ci(run_results):
    # The input 'run_results' is a list of dictionaries (raw run results).
    scores = []
    for result in run_results:
        if "dictResult" in result:
            # Safely get the 'grounding_score' from inside the 'dictResult' dictionary
            score = result["dictResult"].get("grounding_score", None)
            scores.append(score)

    df = pd.DataFrame({"grounding_score": scores})

    if len(df) > 0:
        mean_score = float(df.grounding_score.mean())
    else:
        mean_score = 0.0

    return mean_score


# In[14]:


response_string='```json\n{"sentence": "The proposed Temporal-Patchify encoding method achieves state-of-the-art performance, with a higher pAUC and a significantly lower false-alarm burden compared to the baseline Temporal-Tile SegFormer, while maintaining clinically usable sensitivity.", "label": "supported", "rationale": "The sentence is explicitly supported by the context, which states that the Temporal-Patchify method achieves state-of-the-art performance, has a pAUC 16.2% higher and a false-alarm burden 44.4% lower than the baseline Temporal-Tile SegFormer, and maintains clinically usable sensitivity.", "excerpt": "Our proposed Temporal-Patchify encoding method achieves state-of-the-art performance.; We achieved 0.61 pAUC, which is 16.2\\\\% higher than the baseline Temporal-Tile SegFormer of Parani et al.; The false-alarm burden (0.40{\\\\textpm}0.28 FA/h) is 44.4\\\\% lower than the Temporal-Tile SegFormer baseline while maintaining clinically usable sensitivity"}\n{"sentence": "Its performance exceeds chance, and it also demonstrates faster end-to-end inference throughput.", "label": "supported", "rationale": "The context directly supports this by stating that statistical validation confirmed the performance exceeds chance. It also notes an end-to-end inference throughput that exceeds SegFormer by over 20%, confirming faster inference speed.", "excerpt": "confirming performance exceeds chance.; Finally, we report end-to-end inference through-put up to 920 windows/s, confirming MambaVision{\\\\textquoteright}s fastest inference speed, exceeding SegFormer by over 20\\\\%."}\n```'
def extract_labels_from_response(response_string):
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
#problems with old parser: it was not able to parse the response correctly and extract the labels, which are crucial for calculating the grounding score. The new parser is designed to be more robust and can handle various formatting issues in the model's response, ensuring that we can accurately extract the necessary information for evaluation.
#extract_labels_from_response(response_string),parse_ufg_rev21_verdict(response_string,judge='x', verbose=True)


# In[22]:


from statistics import mean, stdev
def calculate_score(run_results):
    # The input 'run_results' is a list of dictionaries (raw run results).
    scoring = {}

    for result in run_results:
        #for debuugi
        if "dictResult" in result:
            sample_id=result["dictResult"]["sample_id"]#should be 'sample_id' in future
            scoring[sample_id]={}
            #from result["dictResult"] get values for "quality-<judge>-rating" where 'judge' can be one ot all judges in JUDGES list and store them in a list of strings in the format "<judge>: <rating>"
            scoring[sample_id]['quality']={}
            scoring[sample_id]['grounding']={}
            for judge in JUDGES:
                rating_key = f"quality-{judge}-rating"

                if rating_key in result["dictResult"]:
                    rating_value = labels_completness.get(result["dictResult"][rating_key], None)
                    scoring[sample_id]['quality'][judge] = rating_value
                grounding_key = f"grounding-{judge}-text"
                if grounding_key in result["dictResult"]:# and len(result["dictResult"][grounding_key]) > 0
                    grounding_value = result["dictResult"][grounding_key]
                    #
                    #try first simple parser to extract labels from grounding_value, if it fails, then use the more complex parser which also provides rationales and excerpts. The reason for this is that the simple parser is more robust to formatting issues in the model response and can still extract the necessary information for scoring, while the complex parser may fail if the formatting is not exactly as expected.
                    parsed_answers = extract_labels_from_response(grounding_value)

                    if len(parsed_answers) == 0 or (not all([labels_grounding[x] for x in parsed_answers if x in labels_grounding])):
                        print(f"no labels with simple parser for grounding evaluation for sample_id {sample_id} {grounding_key}")
                        _, parsed_answers = parse_ufg_rev21_verdict(grounding_value, judge, verbose=False)
                        parsed_answers=[parsed_answer["label"] for parsed_answer in parsed_answers]

                    if len(parsed_answers) > 0:
                        scoring[sample_id]['grounding'][judge] = mean(filter(lambda x: x is not None, [labels_grounding.get(x, None) for x in parsed_answers]))
                    else:
                        print(f"no labels were extracted for grounding evaluation for sample_id:{sample_id} {grounding_key}:{grounding_value}")
                        scoring[sample_id]['grounding'][judge] = None

    #save scoring to json file
    with open(file_score, "w") as f:
        json.dump(scoring, f, indent=4)
    scores=[sum([mean([z for z in y.values() if z is not None]) for y in x.values() if len(y.values()) > 0])/sample_score_max for x in scoring.values()]   
    return mean(filter(lambda x: x is not None, scores)), stdev(filter(lambda x: x is not None, scores))


# In[16]:


import json
import os
# Calculate accuracy and confidence intervals
run_files = kbench.kaggle.serialization.get_runs_filenames(runs)

# 1. Patch the files
for file_path in run_files:
    with open(file_path, 'r') as f:
        run_data = json.load(f)

    # If modelVersion is empty or missing, inject a dummy slug
    if not run_data.get("modelVersion"):
        run_data["modelVersion"] = {"slug": model_slug}
    elif "slug" not in run_data["modelVersion"]:
        run_data["modelVersion"]["slug"] = model_slug

    with open(file_path, 'w') as f:
        json.dump(run_data, f)


# In[19]:


#local debugging
if 'project' in os.getcwd():
    id0="10.1016/j.jstrokecerebrovasdis.2026.108586"
    os.chdir("out/pubmed-clinical-case-conclusions/20260413_1800");
    run_files = [f for f in os.listdir() if f.endswith(".run.json") and not f.endswith("Run_aggregated.run.json")]
    print(run_files)


# In[23]:


# Aggregate run results
kbench.kaggle.serialization.merge_results_from_runfiles(
    run_files=run_files, aggregate_fn=calculate_score, delete_run_files=False
)


# ## View Aggregated Results
# 
# In the aggregated runfile, the "results" field will store the mean grounding score for the model.

# In[24]:


get_ipython().system('cat *-Run_aggregated.run.json')


# In[ ]:


#!cat tasks_scoring.json

