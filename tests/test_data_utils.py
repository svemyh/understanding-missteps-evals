"""
Tests for the data utilities.
"""
import os
import json
import tempfile
import pytest
from src.utils.data_utils import (
    load_math_problems,
    format_problem_for_llm,
    extract_problem_statement,
    extract_golden_answer,
    extract_solution
)


def test_load_math_problems():
    """
    Test loading math problems from a JSON file.
    """
    # Create a temporary JSON file with some test data
    test_data = {
        "entries": [
            {
                "id": 1,
                "problem_statement": "Test problem 1",
                "golden_answer": "Test answer 1",
                "solution": "Test solution 1"
            },
            {
                "id": 2,
                "problem_statement": "Test problem 2",
                "golden_answer": "Test answer 2",
                "solution": None
            },
            {
                "id": 3,
                "problem_statement": "Test problem 3",
                "golden_answer": "No official answer.",
                "solution": "Test solution 3"
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(test_data, temp_file)
        temp_file_path = temp_file.name
    
    try:
        # Load the math problems
        problems = load_math_problems(temp_file_path)
        
        # Check that only valid problems are loaded
        assert len(problems) == 2, f"Expected 2 problems, but got {len(problems)}"
        assert problems[0]["id"] == 1
        assert problems[1]["id"] == 2
        
    finally:
        # Delete the temporary file
        os.unlink(temp_file_path)


def test_format_problem_for_llm():
    """
    Test formatting a problem for an LLM.
    """
    problem = {
        "problem_statement": "What is 2 + 2?",
        "golden_answer": "4",
        "solution": "2 + 2 = 4"
    }
    
    formatted = format_problem_for_llm(problem)
    
    assert "Problem: What is 2 + 2?" in formatted
    assert "Provide a concise mathematical answer" in formatted


def test_extract_functions():
    """
    Test the extract functions.
    """
    problem = {
        "problem_statement": "What is 2 + 2?",
        "golden_answer": "4",
        "solution": "2 + 2 = 4"
    }
    
    assert extract_problem_statement(problem) == "What is 2 + 2?"
    assert extract_golden_answer(problem) == "4"
    assert extract_solution(problem) == "2 + 2 = 4"
    
    # Test with missing fields
    empty_problem = {}
    assert extract_problem_statement(empty_problem) == ""
    assert extract_golden_answer(empty_problem) == ""
    assert extract_solution(empty_problem) is None