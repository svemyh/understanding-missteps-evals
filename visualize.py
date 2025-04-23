"""
Visualization script for mathematical correctness evaluation results.
"""
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
from typing import Dict, List, Any, Optional


def parse_results(results_file: str) -> List[float]:
    """
    Parse the test results JSON file and extract the scores.
    
    Args:
        results_file: Path to the JSON file containing evaluation results
        
    Returns:
        List of scores from the test results
    """
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    scores = []
    for test_result_str in data.get('test_results', []):
        # Use regex to extract the score from the string representation
        score_match = re.search(r'score=(\d+\.\d+)', test_result_str)
        if score_match:
            score = float(score_match.group(1))
            scores.append(score)
    
    return scores


def plot_scores(scores: List[float], output_path: Optional[str] = None):
    """
    Create a simple plot of scores against a counter (1,2,3,4).
    
    Args:
        scores: List of scores to plot
        output_path: Optional path to save the plot image
    """
    # Create counters for x-axis
    counters = list(range(1, len(scores) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(counters, scores, 'bo-', linewidth=2, markersize=8)
    plt.scatter(counters, scores, color='blue', s=100)
    
    # Add labels for each point
    for i, score in enumerate(scores):
        plt.annotate(f'{score:.2f}', 
                    (counters[i], score),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
    
    # Set up the plot
    plt.xlabel('Test Case Number')
    plt.ylabel('Score')
    plt.title('Math Answer Correctness Scores by Test Case')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(counters)
    plt.ylim(-0.05, 1.05)  # Assuming scores are between 0 and 1
    
    # Add a horizontal line at threshold 0.7 if mentioned in the results
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='Threshold (0.7)')
    plt.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    """
    Main entry point for the visualization script.
    """
    parser = argparse.ArgumentParser(
        description='Visualize evaluation results from mathematical correctness tests.'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the evaluation results JSON file.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save the plot image. If not provided, the plot is displayed.'
    )
    
    args = parser.parse_args()
    
    # Parse the results and extract scores
    scores = parse_results(args.input)
    
    # Plot the scores
    plot_scores(scores, args.output)


if __name__ == '__main__':
    main()