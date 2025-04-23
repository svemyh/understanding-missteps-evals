"""
Custom DeepEval metrics for evaluating mathematical correctness.
"""
from typing import Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from src.utils.llm_client import judge_mathematical_correctness


class MathAnswerCorrectnessMetric(BaseMetric):
    """
    A custom metric that evaluates whether the final answer in an LLM's response
    matches the expected golden answer for mathematical problems.
    
    The metric uses an LLM as a judge to evaluate mathematical equivalence,
    allowing for different expressions of the same mathematical concept.
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        model: Optional[str] = None,
        include_reason: bool = True,
        strict_mode: bool = False,
        async_mode: bool = True,
        verbose_mode: bool = False
    ):
        """
        Initialize the MathAnswerCorrectnessMetric.
        
        Args:
            threshold: The threshold for the metric to be considered successful.
            model: The LLM model to use as a judge.
            include_reason: Whether to include a reason in the metric result.
            strict_mode: Whether to enforce a strict comparison.
            async_mode: Whether to enable asynchronous execution.
            verbose_mode: Whether to enable verbose mode for debugging.
        """
        self.threshold = threshold
        self.evaluation_model = model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.verbose_mode = verbose_mode
        self.score = 0.0
        self.reason = ""
        self.success = False
        self.error = None
        
    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measure the mathematical correctness of an LLM's answer.
        
        Args:
            test_case: The test case containing the problem, expected output,
                      and actual output.
        
        Returns:
            The score representing the mathematical correctness.
        """
        try:
            # Extract problem, actual output, and expected output from test case
            problem = test_case.input
            actual_output = test_case.actual_output
            expected_output = test_case.expected_output
            
            if self.verbose_mode:
                print(f"Problem: {problem}")
                print(f"Actual Output: {actual_output}")
                print(f"Expected Output: {expected_output}")
            
            # Use LLM judge to evaluate mathematical correctness
            judgment = judge_mathematical_correctness(
                problem=problem,
                student_answer=actual_output,
                expected_answer=expected_output,
                model=self.evaluation_model
            )
            
            self.score = judgment['score']
            self.reason = judgment['reason']
            
            # Apply strict mode if enabled
            if self.strict_mode and self.score < 1.0:
                self.score = 0.0
                
            # Determine success based on threshold
            self.success = self.score >= self.threshold
            
            if self.verbose_mode:
                print(f"Score: {self.score}")
                print(f"Reason: {self.reason}")
                print(f"Success: {self.success}")
                
            return self.score
            
        except Exception as e:
            self.error = str(e)
            self.success = False
            raise
    
    async def a_measure(self, test_case: LLMTestCase) -> float:
        """
        Asynchronous version of the measure method.
        
        Since we don't have an async version of the judge_mathematical_correctness
        function, we simply call the synchronous version.
        
        Args:
            test_case: The test case to evaluate.
            
        Returns:
            The score representing the mathematical correctness.
        """
        return self.measure(test_case)
    
    def is_successful(self) -> bool:
        """
        Check if the metric was successful.
        
        Returns:
            True if the score is above the threshold, False otherwise.
        """
        if self.error is not None:
            self.success = False
        return self.success
    
    @property
    def __name__(self):
        """
        Get the name of the metric.
        
        Returns:
            The name of the metric.
        """
        return "Math Answer Correctness Metric"