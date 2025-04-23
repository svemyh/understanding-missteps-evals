"""
Utilities for loading and processing mathematical problem data.
"""
import json
from typing import Dict, List, Optional, Any


def load_math_problems(file_path: str) -> List[Dict[str, Any]]:
    """
    Load mathematical problems from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing the problems.
        
    Returns:
        A list of dictionaries containing the problem entries.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Filter out entries without a golden answer or with placeholder answers
    entries = data.get('entries', [])
    valid_entries = [
        entry for entry in entries 
        if entry.get('golden_answer') and 
        entry.get('golden_answer') != "No official answer." and
        entry.get('problem_statement')
    ]
    
    return valid_entries


def format_problem_for_llm(problem: Dict[str, Any]) -> str:
    """
    Format a mathematical problem for an LLM.
    
    Args:
        problem: Dictionary containing the problem data.
        
    Returns:
        A string with the formatted problem.
    """
    return (
        f"Problem: {problem['problem_statement']}\n\n"
        f"Provide a concise mathematical answer to this problem."
    )


def extract_problem_statement(problem: Dict[str, Any]) -> str:
    """
    Extract the problem statement from a problem dictionary.
    
    Args:
        problem: Dictionary containing the problem data.
        
    Returns:
        The problem statement as a string.
    """
    return problem.get('problem_statement', '')


def extract_golden_answer(problem: Dict[str, Any]) -> str:
    """
    Extract the golden answer from a problem dictionary.
    
    Args:
        problem: Dictionary containing the problem data.
        
    Returns:
        The golden answer as a string.
    """
    return problem.get('golden_answer', '')


def extract_solution(problem: Dict[str, Any]) -> Optional[str]:
    """
    Extract the solution (if available) from a problem dictionary.
    
    Args:
        problem: Dictionary containing the problem data.
        
    Returns:
        The solution as a string, or None if not available.
    """
    return problem.get('solution')