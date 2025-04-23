"""
Tests for the evaluation module.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.evaluation import (
    create_test_cases,
    evaluate_single_problem
)


@patch('src.evaluation.generate_solution')
def test_create_test_cases(mock_generate_solution):
    """
    Test creating test cases from mathematical problems.
    """
    # Mock the generate_solution function
    mock_generate_solution.return_value = "Generated solution"
    
    # Create some sample problems
    problems = [
        {
            "id": 1,
            "problem_statement": "Problem 1",
            "golden_answer": "Answer 1"
        },
        {
            "id": 2,
            "problem_statement": "Problem 2",
            "golden_answer": "Answer 2"
        }
    ]
    
    # Create test cases without generating solutions
    test_cases = create_test_cases(problems, generate_solutions=False)
    
    # Check the test cases
    assert len(test_cases) == 2
    assert test_cases[0].input == "Problem 1"
    assert test_cases[0].expected_output == "Answer 1"
    assert "No solution generated" in test_cases[0].actual_output
    assert not mock_generate_solution.called
    
    # Create test cases with generating solutions
    test_cases = create_test_cases(problems, generate_solutions=True)
    
    # Check the test cases
    assert len(test_cases) == 2
    assert test_cases[0].input == "Problem 1"
    assert test_cases[0].expected_output == "Answer 1"
    assert test_cases[0].actual_output == "Generated solution"
    assert mock_generate_solution.call_count == 2


@patch('src.evaluation.MathAnswerCorrectnessMetric')
def test_evaluate_single_problem(mock_metric_class):
    """
    Test evaluating a single problem.
    """
    # Create a mock metric instance
    mock_metric = MagicMock()
    mock_metric.measure.return_value = 0.85
    mock_metric.reason = "The answers are mathematically equivalent."
    mock_metric.is_successful.return_value = True
    
    # Make the mock class return the mock instance
    mock_metric_class.return_value = mock_metric
    
    # Evaluate a single problem
    score, reason, success = evaluate_single_problem(
        problem_statement="What is 2 + 2?",
        llm_answer="4",
        golden_answer="4",
        verbose=True
    )
    
    # Check the results
    assert score == 0.85
    assert reason == "The answers are mathematically equivalent."
    assert success is True
    
    # Check that the metric was called with the correct test case
    mock_metric_class.assert_called_once()
    mock_metric.measure.assert_called_once()
    test_case = mock_metric.measure.call_args[0][0]
    assert test_case.input == "What is 2 + 2?"
    assert test_case.actual_output == "4"
    assert test_case.expected_output == "4"