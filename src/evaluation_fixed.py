"""
Main evaluation pipeline for evaluating mathematical correctness.
"""
import os
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from src.utils.data_utils import (
    load_math_problems,
    format_problem_for_llm,
    extract_problem_statement,
    extract_golden_answer
)
from src.utils.llm_client import generate_solution
from src.custom_metrics import MathAnswerCorrectnessMetric


# Load environment variables
load_dotenv()


def create_test_cases(
    problems: List[Dict[str, Any]],
    generate_solutions: bool = False,
    solution_model: Optional[str] = None
) -> List[Tuple[LLMTestCase, int]]:
    """
    Create test cases from mathematical problems.
    
    Args:
        problems: List of problem dictionaries.
        generate_solutions: Whether to generate solutions using an LLM.
        solution_model: The model to use for generating solutions.
        
    Returns:
        A list of tuples containing LLMTestCase objects and their original IDs.
    """
    test_cases = []
    
    for problem in problems:
        # Extract problem components
        problem_statement = extract_problem_statement(problem)
        golden_answer = extract_golden_answer(problem)
        problem_id = problem.get("id")  # Get the original ID
        
        # Format problem for LLM
        llm_formatted_problem = format_problem_for_llm(problem)
        
        # Generate a solution if required
        if generate_solutions:
            actual_output = generate_solution(
                llm_formatted_problem,
                model=solution_model
            )
        else:
            # Use a placeholder actual output if not generating solutions
            actual_output = "No solution generated. Enable solution generation to test with actual LLM outputs."
        
        # Create a test case
        test_case = LLMTestCase(
            input=problem_statement,
            actual_output=actual_output,
            expected_output=golden_answer
        )
        
        # Store test case along with its original ID
        test_cases.append((test_case, problem_id))
    
    return test_cases


def run_evaluation(
    data_path: str,
    generate_solutions: bool = False,
    solution_model: Optional[str] = None,
    judge_model: Optional[str] = None,
    threshold: float = 0.7,
    strict_mode: bool = False,
    verbose: bool = False,
    log_to_confident: bool = False,
    use_deepeval: bool = True
) -> Dict[str, Any]:
    """
    Run an evaluation on mathematical problems.
    
    Args:
        data_path: Path to the data file.
        generate_solutions: Whether to generate solutions using an LLM.
        solution_model: The model to use for generating solutions.
        judge_model: The model to use for judging solutions.
        threshold: The threshold for a metric to be considered successful.
        strict_mode: Whether to use strict mode for evaluation.
        verbose: Whether to enable verbose mode.
        log_to_confident: Whether to log results to Confident AI.
        use_deepeval: Whether to use DeepEval framework.
        
    Returns:
        A dictionary containing the evaluation results.
    """
    # Load problems
    problems = load_math_problems(data_path)
    
    if verbose:
        print(f"Loaded {len(problems)} problems.")
    
    # Create test cases
    test_cases_with_ids = create_test_cases(
        problems=problems,
        generate_solutions=generate_solutions,
        solution_model=solution_model
    )
    
    # Extract just the test cases for DeepEval (keeping ID mapping for later)
    test_cases = [tc for tc, _ in test_cases_with_ids]
    problem_ids = [pid for _, pid in test_cases_with_ids]
    
    if verbose:
        print(f"Created {len(test_cases)} test cases.")
    
    # Create the metric
    math_correctness_metric = MathAnswerCorrectnessMetric(
        threshold=threshold,
        model=judge_model,
        strict_mode=strict_mode,
        verbose_mode=verbose
    )
    
    # Run the evaluation
    if verbose:
        print("Running evaluation...")
        
    if use_deepeval:
        # Set up evaluation parameters for DeepEval
        eval_params = {
            "test_cases": test_cases,
            "metrics": [math_correctness_metric],
            "print_results": True,
            # DeepEval requires 'model' and 'prompt template' keys in hyperparameters (note the space!)
            "hyperparameters": {
                "model": solution_model or os.getenv("SOLUTION_MODEL", "openai:gpt-4o"),
                "prompt template": "Mathematical problem-solving with LLM-as-judge evaluation"
            }
        }
        
        # Add additional hyperparameters if logging to Confident AI
        if log_to_confident:
            eval_params["hyperparameters"].update({
                "judge_model": judge_model or os.getenv("JUDGE_MODEL", "openai:gpt-4o"),
                "threshold": threshold,
                "strict_mode": strict_mode
            })
        
        # Run the evaluation using DeepEval
        deepeval_results = evaluate(**eval_params)
        
        # Add problem IDs to the results
        results = deepeval_results
        if "test_cases" in results:
            for i, test_case_result in enumerate(results["test_cases"]):
                if i < len(problem_ids):
                    test_case_result["id"] = problem_ids[i]
    else:
        # Run a manual evaluation without DeepEval
        results = {
            "test_cases": [],
            "metrics": [math_correctness_metric.__name__],
            "summary": {
                "total": len(test_cases),
                "passed": 0,
                "failed": 0
            }
        }
        
        for i, test_case in enumerate(test_cases):
            if verbose:
                print(f"\nEvaluating test case {i+1}/{len(test_cases)}")
                print(f"Problem: {test_case.input}")
                print(f"LLM Answer: {test_case.actual_output}")
                print(f"Golden Answer: {test_case.expected_output}")
            
            score = math_correctness_metric.measure(test_case)
            success = math_correctness_metric.is_successful()
            reason = math_correctness_metric.reason
            
            results["test_cases"].append({
                "id": problem_ids[i],  # Add the original problem ID
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "expected_output": test_case.expected_output,
                "score": score,
                "success": success,
                "reason": reason
            })
            
            if success:
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
                
            if verbose:
                print(f"Score: {score}")
                print(f"Success: {success}")
                print(f"Reason: {reason}")
                print("-" * 50)
    
    return results


def evaluate_single_problem(
    problem_statement: str,
    llm_answer: str,
    golden_answer: str,
    judge_model: Optional[str] = None,
    threshold: float = 0.7,
    verbose: bool = False
) -> Tuple[float, str, bool]:
    """
    Evaluate a single mathematical problem.
    
    Args:
        problem_statement: The problem statement.
        llm_answer: The answer generated by an LLM.
        golden_answer: The golden answer to compare against.
        judge_model: The model to use for judging.
        threshold: The threshold for a metric to be considered successful.
        verbose: Whether to enable verbose mode.
        
    Returns:
        A tuple containing the score, reason, and success status.
    """
    # Create a test case
    test_case = LLMTestCase(
        input=problem_statement,
        actual_output=llm_answer,
        expected_output=golden_answer
    )
    
    # Create the metric
    math_correctness_metric = MathAnswerCorrectnessMetric(
        threshold=threshold,
        model=judge_model,
        verbose_mode=verbose
    )
    
    # Measure the metric
    score = math_correctness_metric.measure(test_case)
    
    return score, math_correctness_metric.reason, math_correctness_metric.is_successful()