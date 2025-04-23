"""
Tests for the custom metrics.
"""
import pytest
from deepeval.test_case import LLMTestCase
from src.custom_metrics import MathAnswerCorrectnessMetric


def test_math_answer_correctness_metric_identical():
    """
    Test the MathAnswerCorrectnessMetric with identical answers.
    """
    # Create a test case with identical actual and expected outputs
    test_case = LLMTestCase(
        input="If f_i(x) is a convex function for all i=0,...,m, then what can we say about g(λ)?",
        actual_output="g(λ) is concave in λ.",
        expected_output="g(λ) is concave in λ."
    )
    
    # Create the metric
    metric = MathAnswerCorrectnessMetric(threshold=0.7)
    
    # Measure the metric
    score = metric.measure(test_case)
    
    # Check the score
    assert score >= 0.9, f"Score should be >= 0.9, but got {score}"
    assert metric.is_successful()


def test_math_answer_correctness_metric_equivalent():
    """
    Test the MathAnswerCorrectnessMetric with equivalent but different answers.
    """
    # Create a test case with equivalent but different answers
    test_case = LLMTestCase(
        input="If f_i(x) is a convex function for all i=0,...,m, then what can we say about g(λ)?",
        actual_output="The function g(λ) is concave with respect to λ.",
        expected_output="g(λ) is concave in λ."
    )
    
    # Create the metric
    metric = MathAnswerCorrectnessMetric(threshold=0.7)
    
    # Measure the metric
    score = metric.measure(test_case)
    
    # Check the score
    assert score >= 0.8, f"Score should be >= 0.8, but got {score}"
    assert metric.is_successful()


def test_math_answer_correctness_metric_incorrect():
    """
    Test the MathAnswerCorrectnessMetric with incorrect answers.
    """
    # Create a test case with incorrect answer
    test_case = LLMTestCase(
        input="If f_i(x) is a convex function for all i=0,...,m, then what can we say about g(λ)?",
        actual_output="g(λ) is convex in λ.",
        expected_output="g(λ) is concave in λ."
    )
    
    # Create the metric
    metric = MathAnswerCorrectnessMetric(threshold=0.7)
    
    # Measure the metric
    score = metric.measure(test_case)
    
    # Check the score
    assert score < 0.5, f"Score should be < 0.5, but got {score}"
    assert not metric.is_successful()


def test_math_answer_correctness_metric_partially_correct():
    """
    Test the MathAnswerCorrectnessMetric with partially correct answers.
    """
    # Create a test case with partially correct answer
    test_case = LLMTestCase(
        input="If f_i(x) is a convex function for all i=0,...,m, then what can we say about g(λ)?",
        actual_output="g(λ) has some properties, it might be concave.",
        expected_output="g(λ) is concave in λ."
    )
    
    # Create the metric
    metric = MathAnswerCorrectnessMetric(threshold=0.7)
    
    # Measure the metric
    score = metric.measure(test_case)
    
    # Check the score - we expect a medium score for partially correct answers
    assert 0.3 <= score <= 0.8, f"Score should be between 0.3 and 0.8, but got {score}"