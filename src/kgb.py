import kaggle_benchmarks as kbench
from kaggle_benchmarks.prompting import ResponseParsingError
from pydantic import BaseModel, Field

class BinaryQA(BaseModel):
    question: str = Field(description="The generated binary question.")
    answer: str = Field(description="The answer to the question, must be one of 'Yes', 'No', or 'Unknown'.")

@kbench.task(name="rewrite_conclusion_as_binary_qa")
def rewrite_conclusion_as_binary_qa(llm,conclusion = "This two-stage IPD meta-analysis provides robust evidence of a bidirectional relationship between CLD and OP in older adults. These findings underscore the need for integrated screening and management of both conditions in aging populations."
):
    
    prompt = f"""
    Please analyze the following scientific conclusion and rephrase it as a single, clear binary question and a one-word answer from the options: Yes, No, or Unknown.

    Conclusion: "{conclusion}"

    Your output must be a JSON object with two keys: "question" and "answer".
    """

    try:
        response = llm.prompt(prompt, schema=BinaryQA)

        kbench.assertions.assert_contains_regex(
            r"(?i)^(yes|no|unknown)",
            response.answer.strip(),
           # expectation="The answer should be 'Yes' based on the provided text."
        )

    except ResponseParsingError as e:
        kbench.assertions.assert_fail(
            expectation=f"The model's output did not conform to the BinaryQA schema. Error: {e.error}"
        )

