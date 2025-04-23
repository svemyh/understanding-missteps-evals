# Understanding Missteps: Math Answer Evaluation

This project implements an LLM-as-a-judge system using DeepEval to evaluate the mathematical correctness of LLM answers. It compares an LLM's answer against a golden answer from a dataset of mathematical problems.

## Setup

1. Clone the repository

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Rename a `.env.example` to `.env` and add your API keys


## Usage

### Get started

```bash
python example.py
```

### Evaluating mathematical problems

```bash
python main.py \
--data-path data/latest-all-eecs127at-shortened.json \
--generate-solutions \
--verbose \
--judge-model openai:gpt-4o \
--solution-model openai:gpt-4o
```

### Visualizing results

```bash
python visualize.py --input results.json
```
# understanding-missteps-evals
