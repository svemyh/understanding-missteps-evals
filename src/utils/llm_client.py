"""
Utilities for interacting with LLMs using AISuite.
"""
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import aisuite as ai


# Load environment variables
load_dotenv()


def get_client():
    """
    Get an AISuite client instance.
    
    Returns:
        An AISuite client instance.
    """
    return ai.Client()


def generate_solution(
    problem_text: str, 
    model: Optional[str] = None
) -> str:
    """
    Generate a solution to a mathematical problem using an LLM.
    
    Args:
        problem_text: The text of the problem to solve.
        model: The model identifier to use (format: provider:model_name).
              If None, uses the SOLUTION_MODEL environment variable.
              
    Returns:
        The generated solution as a string.
    """
    client = get_client()
    model = model or os.getenv("SOLUTION_MODEL", "openai:gpt-4o")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert mathematician. Solve the given mathematical problem "
                "and provide a clear, concise answer. Include only the final answer, "
                "not the steps to get there."
            )
        },
        {"role": "user", "content": problem_text}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,  # Use deterministic output for consistency
    )
    
    return response.choices[0].message.content


def judge_mathematical_correctness(
    problem: str,
    student_answer: str,
    expected_answer: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use an LLM to judge whether a student's answer is mathematically correct.
    
    Args:
        problem: The text of the problem.
        student_answer: The answer provided by the student (or LLM).
        expected_answer: The expected (golden) answer.
        model: The model identifier to use (format: provider:model_name).
               If None, uses the JUDGE_MODEL environment variable.
    
    Returns:
        A dictionary containing the judgment result with 'score' and 'reason'.
    """
    client = get_client()
    model = model or os.getenv("JUDGE_MODEL", "openai:gpt-4o")
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a mathematics professor evaluating a student's answer "
                "to determine if it's mathematically correct. Evaluate only the "
                "mathematical correctness, not the formatting, grammar, or style. "
                "Two answers are equivalent if they represent the same mathematical "
                "concept, even if expressed differently. For example, '1/2', '0.5', "
                "and '50%' are equivalent. Similarly, 'x^2 + 2x + 1' and '(x+1)^2' "
                "are equivalent. Ignore superficial differences like spacing, "
                "capitalization, or the use of different but equivalent mathematical "
                "notation. Finally, if one answer is just a transformation or simplification "
                "of the other, they are equivalent.\n\n"
                "At the end of your analysis, provide a score from 0 to 1, where:\n"
                "- 1: The student's answer is mathematically equivalent to the expected answer\n"
                "- 0.5-0.9: The student's answer is partially correct or on the right track\n"
                "- 0: The student's answer is incorrect\n\n"
                "Report your analysis and score in JSON format with 'score' (a number) "
                "and 'reason' (string) fields."
            )
        },
        {
            "role": "user",
            "content": (
                f"Problem: {problem}\n\n"
                f"Student's Answer: {student_answer}\n\n"
                f"Expected Answer: {expected_answer}\n\n"
                "Is the student's answer mathematically correct/equivalent to the expected answer? "
                "Analyze in detail, looking for mathematical equivalence rather than exact string matches."
            )
        }
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,  # Use deterministic output for consistency
        response_format={"type": "json_object"},
    )
    
    # Parse JSON response
    try:
        import json
        content = response.choices[0].message.content
        result = json.loads(content)
        
        # Ensure the expected fields are present
        if 'score' not in result or 'reason' not in result:
            # Set default values if expected fields are missing
            if 'score' not in result:
                result['score'] = 0.0
            if 'reason' not in result:
                result['reason'] = "Error: Judge did not provide a reason."
                
        # Ensure score is a float
        result['score'] = float(result['score'])
        
        return result
    except Exception as e:
        # Return a default response in case of parsing error
        return {
            'score': 0.0,
            'reason': f"Error parsing judge response: {str(e)}\nRaw response: {response.choices[0].message.content}"
        }