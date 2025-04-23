"""
Main entry point for the evaluation pipeline.
"""
import argparse
from dotenv import load_dotenv
from src.evaluation_fixed import run_evaluation


# Load environment variables
load_dotenv()


def main():
    """
    Main entry point for the evaluation pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate LLM answers to mathematical problems using an LLM as a judge.'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/latest-all-eecs127at.json',
        help='Path to the data file containing the mathematical problems.'
    )
    
    parser.add_argument(
        '--generate-solutions',
        action='store_true',
        help='Whether to generate solutions using an LLM.'
    )
    
    parser.add_argument(
        '--solution-model',
        type=str,
        help='The model to use for generating solutions. Format: provider:model_name'
    )
    
    parser.add_argument(
        '--judge-model',
        type=str,
        help='The model to use for judging solutions. Format: provider:model_name'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.7,
        help='The threshold for a metric to be considered successful.'
    )
    
    parser.add_argument(
        '--strict-mode',
        action='store_true',
        help='Whether to use strict mode for evaluation.'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether to enable verbose mode.'
    )
    
    parser.add_argument(
        '--log-to-confident',
        action='store_true',
        help='Whether to log results to Confident AI.'
    )
    
    parser.add_argument(
        '--no-deepeval',
        action='store_true',
        help='Run without using DeepEval framework (manual evaluation).'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='results.json',
        help='Path where the evaluation results will be saved.'
    )
    
    args = parser.parse_args()
    
    # Run the evaluation
    results = run_evaluation(
        data_path=args.data_path,
        generate_solutions=args.generate_solutions,
        solution_model=args.solution_model,
        judge_model=args.judge_model,
        threshold=args.threshold,
        strict_mode=args.strict_mode,
        verbose=args.verbose,
        log_to_confident=args.log_to_confident,
        use_deepeval=not args.no_deepeval
    )

    # Save results with the specified output path
    save_results(results, output_path=args.output_path)
    
    return results


def save_results(results, output_path='results.json'):
    """
    Save the evaluation results to a specified file.

    :param results: The results data to save. Each test case result should contain
                   the original problem ID from the dataset.
    :param output_path: The path to the output file.
    """
    import json
    import os
    from datetime import datetime
    import inspect
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Convert results to a dictionary if it's not already
    if not isinstance(results, dict):
        # Handle DeepEval's EvaluationResult object
        try:
            # Try to convert the DeepEval result object to a dictionary
            results_dict = {}
            
            # Check for various possible attributes and methods
            if hasattr(results, 'to_dict'):
                # Use the to_dict method if available
                results_dict = results.to_dict()
            elif hasattr(results, '__dict__'):
                # Get the __dict__ attribute
                results_dict = dict(results.__dict__)
            else:
                # Manually extract known attributes
                for attr in dir(results):
                    # Skip private attributes and methods
                    if attr.startswith('_') or callable(getattr(results, attr)):
                        continue
                    try:
                        results_dict[attr] = getattr(results, attr)
                    except:
                        pass
            
            # Add timestamp
            results_dict['timestamp'] = datetime.now().isoformat()
        except Exception as e:
            print(f"Warning: Could not fully convert DeepEval result to dictionary: {e}")
            # Create a minimal dictionary with just the timestamp
            results_dict = {
                'timestamp': datetime.now().isoformat(),
                'conversion_error': str(e),
                'original_type': str(type(results))
            }
    else:
        # For dictionary results (manual evaluation)
        results_dict = results.copy()
        results_dict['timestamp'] = datetime.now().isoformat()
    
    # Verify each test case has an ID field if test_cases exists
    if 'test_cases' in results_dict and isinstance(results_dict['test_cases'], list):
        for test_case in results_dict['test_cases']:
            if isinstance(test_case, dict) and 'id' not in test_case:
                print(f"Warning: Test case missing ID field: {test_case.get('input', '')[:30]}...")
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=4, default=lambda o: str(o))

    print(f"Results saved to {output_path}")
    
    # Print a summary if available
    if 'summary' in results_dict:
        print(f"Summary: Total: {results_dict['summary'].get('total', 0)}, " 
              f"Passed: {results_dict['summary'].get('passed', 0)}, "
              f"Failed: {results_dict['summary'].get('failed', 0)}")
    
    return output_path


if __name__ == '__main__':
    results = main()