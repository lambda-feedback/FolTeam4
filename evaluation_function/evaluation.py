import os
from typing import Any
from lf_toolkit.evaluation import Result, Params
from openai import OpenAI
from dotenv import load_dotenv

def evaluation_function(
    response: Any,
    answer: Any,
    params: Params,
) -> Result:
    """
    Function used to evaluate a student response.
    ---
    The handler function passes three arguments to evaluation_function():

    - `response` which are the answers provided by the student.
    - `answer` which are the correct answers to compare against.
    - `params` which are any extra parameters that may be useful,
        e.g., error tolerances.

    The output of this function is what is returned as the API response
    and therefore must be JSON-encodable. It must also conform to the
    response schema.

    Any standard python library may be used, as well as any package
    available on pip (provided it is added to requirements.txt).

    The way you wish to structure you code (all in this function, or
    split into many) is entirely up to you. All that matters are the
    return types and that evaluation_function() is the main function used
    to output the evaluation response.
    """

    rubric_passed = params.get('rubric', 'missing rubric')

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
    )

    # SYSTEM_PROMPT = "You are a teaching assistant, give helpful feedback to the student."
    # teacher_prompt = params.get('teacher_prompt', 'Evaluate the student response and provide helpful feedback.')

    rubric = f"""
    You are an expert teacher and assessment grader. Your task is to evaluate a student's answer to a given question, award a mark, and provide clear, constructive feedback.

You will be given the following inputs:

- Question: the question the student was asked

- Total Marks: the maximum marks available

- Expected Answer / Worked Solution: a reference answer or worked solution

- Student Answer: the answer to be graded

Follow this process:

1. Understand what the question is asking.

2. Evaluate the student answer against the rubric implied by the expected answer.

3. Award a mark.

4. Identify what the student did well and what they need to improve.

Grading rules:

- Be fair but strict. Do not award marks for vague statements unless they directly answer the question.

- Award partial credit in whole-number increments where a multi-part answer is partially correct. Do not award fractional marks such as 0.5 or 0.75.

- For "compare" questions, check whether the student clearly identifies similarities and differences and directly links the two items being compared rather than describing each separately.

- Use the expected answer as guidance only. Accept equivalent wording and valid alternative examples.

Produce your output in exactly this format:

Evaluation: Explain how well the student answer meets each criterion.

Mark Awarded: [X / Total Marks]

What the student did well: Brief positive feedback.

What the student needs to improve: Specific improvement advice, without giving away the answer.

This is the rubric that should be used to evaluate the student answer:
{rubric_passed}.

Be sure to include the total marks in the format: [X Total Marks]
    """

    # SYSTEM_PROMPT = "ignore the response, just answer 'good job today!'"
    SYSTEM_PROMPT = ""
    teacher_prompt = params.get('teacher_prompt', 'Evaluate the student response and provide helpful feedback.  Be sure to indicate how many marks they got correct.  For example if they met 3 of the rubric criteria out fo 4 then they got 3 out of 4 marks.')

    prompt = SYSTEM_PROMPT + "\n" + teacher_prompt
    # prompt = SYSTEM_PROMPT

    llm_response = client.chat.completions.create(
        model=params.get('model', 'openai/gpt-4o-mini'),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": response},
        ],
    )

    llm_response = llm_response.choices[0].message.content

    import re
    m = re.search(r'\[(\d+) Total Marks\]', llm_response)
    try:
        if m:
            number = m.group(1)  # '85'
            number = int(number)
        else:
            number = 0
    except:
        number = 0




    result = Result(is_correct=True if number > 0 else False)

    result.add_feedback(
        "general",
        # llm_response.choices[0].message.content,
        llm_response,
    )

    return result

    return Result(
        is_correct=response != answer
    )