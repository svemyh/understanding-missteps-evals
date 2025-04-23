"""
Example usage of the mathematical correctness evaluation system.
"""
import os
from dotenv import load_dotenv
from src.evaluation_fixed import evaluate_single_problem
from src.utils.data_utils import load_math_problems, format_problem_for_llm
from src.utils.llm_client import generate_solution


# Load environment variables
load_dotenv()


def main():
    """
    Demonstrate the usage of the evaluation system with a simple example.
    """
    # Check if the file exists
    data_path = 'data/latest-all-eecs127at.json'
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        # Try to find any json file in the data directory
        data_dir = 'data'
        if os.path.exists(data_dir):
            json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            if json_files:
                data_path = os.path.join(data_dir, json_files[0])
                print(f"Using alternative file: {data_path}")
            else:
                print(f"No JSON files found in {data_dir} directory.")
                return
        else:
            print(f"Directory {data_dir} not found.")
            return
            
    print("Loading math problems...")
    problems = load_math_problems(data_path)
    
    # Take the first valid problem as an example
    if not problems:
        print("No valid problems found.")
        return
        
    problem = problems[0]
    problem_statement = problem['problem_statement']
    golden_answer = problem['golden_answer']
    
    print("=" * 50)
    print(f"Problem: {problem_statement}")
    print(f"Golden Answer: {golden_answer}")
    print("=" * 50)
    
    # Format the problem for the LLM
    formatted_problem = format_problem_for_llm(problem)
    
    # Generate a solution using an LLM
    print("Generating solution using LLM...")
    llm_answer = generate_solution(formatted_problem)
    
    print(f"LLM's Answer: {llm_answer}")
    print("=" * 50)
    
    # Evaluate the LLM's answer
    print("Evaluating mathematical correctness...")
    score, reason, success = evaluate_single_problem(
        problem_statement=problem_statement,
        llm_answer=llm_answer,
        golden_answer=golden_answer,
        threshold=0.7,
        verbose=True
    )
    
    print("=" * 50)
    print(f"Score: {score}")
    print(f"Reason: {reason}")
    print(f"Success: {success}")
    print("=" * 50)


if __name__ == '__main__':
    main()