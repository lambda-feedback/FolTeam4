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

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_retries=3,
    )

    # SYSTEM_PROMPT = "You are a teaching assistant, give helpful feedback to the student."
    # teacher_prompt = params.get('teacher_prompt', 'Evaluate the student response and provide helpful feedback.')

    rubric = """
    Question Part a

    1 mark:
    
    Defines a manufacturing business, including the idea of producing tangible goods.
    
    1 mark:
    
    Defines a service business, including the idea of providing intangible services.

 

Question Part b

    1 mark:
    
    Identifies one characteristic of a manufacturing business.
    
    1 mark:
    
    Identifies one related characteristic of a service business.
    
    1 mark:
    
    Makes a clear comparison between manufacturing and service businesses.
    
    1 mark:

    Provides a relevant example to support the comparison.
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

    print(llm_response.choices[0].message.content)

    result = Result(is_correct=True)

    result.add_feedback(
        "general",
        llm_response.choices[0].message.content,
    )

    return result

    return Result(
        is_correct=response != answer
    )