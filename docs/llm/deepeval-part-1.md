Quick Introduction
DeepEval is an open-source evaluation framework for LLMs. DeepEval makes it extremely easy to build and iterate on LLM (applications) and was built with the following principles in mind:

Easily "unit test" LLM outputs in a similar way to Pytest.
Plug-and-use 14+ LLM-evaluated metrics, most with research backing.
Synthetic dataset generation with state-of-the-art evolution techniques.
Metrics are simple to customize and covers all use cases.
Red team, safety scan LLM applications for security vulnerabilities.
Real-time evaluations in production.
Additionally, DeepEval has a cloud platform Confident AI, which allow teams to use DeepEval to evaluate, regression test, red team, and monitor LLM applications on the cloud.

Delivered by

Confident AI
DID YOU KNOW?
If you use Confident AI, you can manage your entire LLM evaluation lifecycle (datasets, testing reports, monitoring, etc.) in one centralized place, and makes sure you do LLM evaluations the right way. Documentation for Confident AI here.

It takes no additional code to setup, is automatically integrated with all code you run using deepeval, and you can click here to sign up for free.

Setup A Python Environment
Go to the root directory of your project and create a virtual environment (if you don't already have one). In the CLI, run:

python3 -m venv venv
source venv/bin/activate

Installation
In your newly created virtual environment, run:

pip install -U deepeval

deepeval runs evaluations locally on your enviornment. To keep your testing reports in a centralized place on the cloud, use Confident AI, the leading evaluation platform for DeepEval:

deepeval login

info
Confident AI is free and allows you to keep all evaluation results on the cloud. Sign up here.

Create Your First Test Case
Run touch test_example.py to create a test file in your root directory. An LLM test case in deepeval is represents a single unit of LLM app interaction. For a series of LLM interactions (i.e. conversation), visit the conversational test cases section instead.

ok

Open test_example.py and paste in your first test case:

test_example.py
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        # Replace this with the actual output of your LLM application
        actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don’t improve in a few days.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
    )
    assert_test(test_case, [correctness_metric])


Run deepeval test run from the root directory of your project:

deepeval test run test_example.py

Congratulations! Your test case should have passed ✅ Let's breakdown what happened.

The variable input mimics a user input, and actual_output is a placeholder for what your application's supposed to output based on this input.
The variable expected_output represents the ideal answer for a given input, and GEval is a research-backed metric provided by deepeval for you to evaluate your LLM output's on any custom metric with human-like accuracy.
In this example, the metric criteria is correctness of the actual_output based on the provided expected_output.
All metric scores range from 0 - 1, which the threshold=0.5 threshold ultimately determines if your test have passed or not.
If you don't have a labelled dataset, don't worry, because the expected_output is only required in this example. Different metrics will require different parameters in an LLMTestCase for evaluation, so be sure to checkout each metric's documentation pages to avoid any unexpected errors.

info
You'll need to set your OPENAI_API_KEY as an enviornment variable before running GEval, since GEval is an LLM-evaluated metric. To use ANY custom LLM of your choice, check out this part of the docs.

Save Results On Confident AI (highly recommended)
Simply login with deepeval login (or click here) to get your API key.

deepeval login

After you've pasted in your API key, Confident AI will generate testing reports for you whenever you run a test run to evaluate your LLM application inside any environment, at any scale, anywhere.

Confident AI
tip
You should save your test run as a dataset on Confident AI, which allows you to reuse the set of inputs and any expected_output, context, etc. for subsequent evaluations.

This allows you to run experiments with different models, prompts, and pinpoint regressions/improvements, and allows for domain experts to collaborate on evaluation datasets that is otherwise difficult for an engineer, researcher, and data scientist to curate.

Save Results Locally
Simply set the DEEPEVAL_RESULTS_FOLDER environment variable to your relative path of choice.

# linux
export DEEPEVAL_RESULTS_FOLDER="./data"

# or windows
set DEEPEVAL_RESULTS_FOLDER=.\data

Run Another Test Run
The whole point of evaluation is to help you iterate towards a better LLM application, and you can do this by comparing the results of two test runs. Simply run another evaluation (we'll use a different actual_output in this example):

test_example.py
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        # Replace this with the actual output of your LLM application
        actual_output="A persistent cough and fever are usually just signs of a cold or flu, so you probably don’t need to worry unless it lasts more than a few weeks. Just rest and drink plenty of fluids, and it should go away on its own.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
    )
    assert_test(test_case, [correctness_metric])


deepeval test run test_example.py

info
In this example, we've delibrately showed a "worse" actual_output that is extremely irrelevant when compared to the previous test case. In reality, you'll not be hardcoding the actual_outputs but rather be generating them at evaluation time.

After running the new test run , you should see the GEval correctness metric failing, and it is because the actual_output is incorrect according to the expected_output. This is actually known as a "regression", and you can catch these by including deepeval in CI/CD pipeliens, or just in a python script.

Based on these two results, you can decide which iterate is better, and whether the latest change you've made to your LLM application is safe to deploy.

Comparing Iterations
Although you can go through hundreds of test cases in your terminal, here's what catching regressions/identifying improvements by comparing test runs looks like on Confident AI (sign up here):

Confident AI
note
We didn't use the same test case data as shown above to demonstrate a more realistic example of what comparing two test runs looks like.

If you look closely, you can see that for the same LLMTestCase (matched by name or input), the difference in its actual_output led to a better Correctness (GEval) metric score.

Green rows mean your LLM improved on this particular test case, while red means it regressed. You'll want to look at the entire test run to see if your results are better or worse!

Create Your First Metric
info
If you're having trouble deciding on which metric to use, you can follow this tutorial or run this command in the CLI:

deepeval recommend metrics

deepeval provides two types of LLM evaluation metrics to evaluate LLM outputs: plug-and-use default metrics, and custom metrics for any evaluation criteria.

Default Metrics
deepeval offers 14+ research backed default metrics covering a wide range of use-cases. Here are a few metrics for RAG pipelines and agents:

Answer Relevancy
Faithfulness
Contextual Relevancy
Contextual Recall
Contextual Precision
Tool Correctness
Task Completion
To create a metric, simply import from the deepeval.metrics module:

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_case = LLMTestCase(input="...", actual_output="...")
relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

relevancy_metric.measure(test_case)
print(relevancy_metric.score, relevancy_metric.reason)

Note that you can run a metric as a standalone or as part of a test run as shown in previous sections.

info
All default metrics are evaluated using LLMs, and you can use ANY LLM of your choice. For more information, visit the metrics introduction section.

Custom Metrics
deepeval provides G-Eval, a state-of-the-art LLM evaluation framework for anyone to create a custom LLM-evaluated metric using natural language. Here's an example:

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

test_case = LLMTestCase(input="...", actual_output="...", expected_output="...")
correctness_metric = GEval(
    name="Correctness",
    criteria="Correctness - determine if the actual output is correct according to the expected output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    strict_mode=True
)

correctness_metric.measure(test_case)
print(correctness_metric.score, correctness_metric.reason)


Under the hood, deepeval first generates a series of evaluation steps, before using these steps in conjuction with information in an LLMTestCase for evaluation. For more information, visit the G-Eval documentation page.

tip
Although GEval is great in many ways as a custom, task-specific metric, it is NOT deterministic. If you're looking for more fine-grained, deterministic control over your metric scores, you should be using the DAGMetric (deep acyclic graph) instead, which is a metric that is deterministic, LLM-powered, and based on a decision tree you define.

Take this decision tree for example, which evaluates a Summarization use case based on the actual_output of your LLMTestCase. Here, we want to check whether the actual_output contains the correct "summary headings", and whether they are in the correct order.

Click to see code associated with diagram below
ok

For more information, visit the DAGMetric documentation.

Measure Multiple Metrics At Once
To avoid redundant code, deepeval offers an easy way to apply as many metrics as you wish for a single test case.

test_example.py
...

def test_everything():
    assert_test(test_case, [correctness_metric, answer_relevancy_metric])

In this scenario, test_everything only passes if all metrics are passing. Run deepeval test run again to see the results:

deepeval test run test_example.py

info
deepeval optimizes evaluation speed by running all metrics for each test case concurrently.

Create Your First Dataset
A dataset in deepeval, or more specifically an evaluation dataset, is simply a collection of LLMTestCases and/or Goldens.

note
A Golden is simply an LLMTestCase with no actual_output, and it is an important concept if you're looking to generate LLM outputs at evaluation time. To learn more about Goldens, click here.

To create a dataset, simply initialize an EvaluationDataset with a list of LLMTestCases or Goldens:

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

first_test_case = LLMTestCase(input="...", actual_output="...")
second_test_case = LLMTestCase(input="...", actual_output="...")

dataset = EvaluationDataset(test_cases=[first_test_case, second_test_case])

Then, using deepeval's Pytest integration, you can utilize the @pytest.mark.parametrize decorator to loop through and evaluate your dataset.

test_dataset.py
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
...

# Loop through test cases using Pytest
@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])

tip
You can also evaluate entire datasets without going through the CLI (if you're in a notebook environment):

from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])

Additionally you can run test cases in parallel by using the optional -n flag followed by a number (that determines the number of processes that will be used) when executing deepeval test run:

deepeval test run test_dataset.py -n 2

Visit the evaluation introduction section to learn about the different types of flags you can use with the deepeval test run command.

Editing Datasets
Especially for those working as part of a team, or have domain experts annotating datasets for you, it is best practice to keep your dataset somewhere as one source of truth. Your team can annotate datasets directly on Confident AI (signup here):

Confident AI
You can then pull the dataset from the cloud to evaluate locally like how you would pull a Github repo.

from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric

dataset = EvaluationDataset()
# supply your dataset alias
dataset.pull(alias="QA Dataset")

evaluate(dataset, metrics=[AnswerRelevancyMetric()])

And you're done! All results will also be available on Confident AI available for comparison and analysis.

Generate Synthetic Datasets
deepeval offers a synthetic data generator that uses state-of-the-art evolution techniques to make synthetic (aka. AI generated) datasets realistic. This is especially helpful if you don't have a prepared evaluation dataset, as it will help you generate the initiate testing data you need to get up and running with evaluation.

caution
You should aim to manually inspect and edit any synthetic data where possible.

Simply supply a list of local document paths to generate a synthetic dataset from your knowledge base.

from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(document_paths=['example.txt', 'example.docx', 'example.pdf'])

dataset = EvaluationDataset(goldens=goldens)


After you're done with generating, simply evaluate your dataset as shown above. Note that deepeval's Synthesizer does NOT generate actual_outputs for each golden. This is because actual_outputs are meant to be generated by your LLM (application), not deepeval's synthesizer.

Visit the synthesizer section to learn how to customize deepeval's synthetic data generation capabilities to your needs.

note
Remember, a Golden is basically an LLMTestCase but with no actual_output.

Red Team Your LLM application
LLM red teaming refers to the process of attacking your LLM application to expose any safety risks it may have, including but not limited to vulnerabilities such as bias, racism, encouraging illegal actions, etc. It is an automated way to test for LLM safety by prompting it with adversarial attacks, which will be all taken care of by deepeval.

info
Red teaming is a different form of testing from what you've seen above because while standard LLM evaluation tests your LLM on its intended functionality, red teaming is meant to test your LLM application against, intentional, adversarial attacks from malicious users.

Here's how you can scan your LLM for vulnerabilities in a few lines of code using deepeval's RedTeamer, an extremely powerful tool to automatically scan for 40+ vulnerabilities. First instantiate a RedTeamer instance:

from deepeval.red_teaming import RedTeamer

red_teamer = RedTeamer(
    # describe purpose of your LLM application
    target_purpose="Provide financial advice related to personal finance and market trends.",
    # supply system prompt template
    target_system_prompt="You are a financial assistant designed to help users with financial planning"
)


Then, supply the target LLM application you wish to scan:

from deepeval.red_teaming import AttackEnhancement, Vulnerability
...

results = red_teamer.scan(
    # your target LLM of type DeepEvalBaseLLM
    target_model=TargetLLM(),
    attacks_per_vulnerability=5,
    vulnerabilities=[v for v in Vulnerability],
    attack_enhancements={
        AttackEnhancement.BASE64: 0.25,
        AttackEnhancement.GRAY_BOX_ATTACK: 0.25,
        AttackEnhancement.JAILBREAK_CRESCENDO: 0.25,
        AttackEnhancement.MULTILINGUAL: 0.25,
    },
)
print("Red Teaming Results: ", results)

deepeval's RedTeamer is highly customizable and offers a range of different advanced red teaming capabilities for anyone to leverage. We highly recommend you read more about the RedTeamer at the red teaming section.

tip
The TargetLLM() you see being provided as argument to the target_model parameter is of type DeepEvalBaseLLM, which is basically a wrapper to wrap your LLM application into deepeval's ecosystem for easy evaluating. You can learn how to create a custom TargetLLM in a few lines of code here.

And that's it! You now know how to not only test your LLM application for its functionality, but also for any underlying risks and vulnerabilities it may expose and make your systems susceptible to malicious attacks.

Using Confident AI
Confident AI is the deepeval cloud platform. While deepeval runs locally and all testing data are lost afterwards, Confident AI offers data persistence, regression testing, sharable testing reports, monitoring, collecting human feedback, and so much more.

note
On-prem hosting is also available. Book a demo to learn more about it.

Here is the LLM development workflow that is highly recommended with Confident AI:

Curate datasets
Run evaluations with dataset
Analyze evaluation results
Improve LLM application based on evaluation results
Run another evaluation on the same dataset
And once your LLM application is live in production, you should:

Monitor LLM outputs, and enable online metrics to flag unsatisfactory outputs
Review unsatisfactory outputs, and decide whether to add it to your evaluation dataset
While there are many LLMOps platform that exist, Confident AI is laser focused on evaluations, although we also offer advanced observability, and native to deepeval, meaning users of deepeval requires no additional code to use Confident AI.

caution
This section is just an overview of Confident AI. If Confident AI sounds interesting, click here for the full Confident AI quickstart guide instead.

Login
Confident AI integrates 100% with deepeval. All you need to do is create an account here, or run the following command to login:

deepeval login

This will open your web browser where you can follow the instructions displayed on the CLI to create an account, get your Confident API key, and paste it in the CLI. You should see a message congratulating your successful login.

tip
You can also login directly in Python once you have your API key:

main.py
deepeval.login_with_confident_api_key("your-confident-api-key")

Curating Datasets
By keeping your datasets on Confident AI, you can ensure that your datasets that are used to run evaluations are always in-sync with your codebase. This is especially helpful if your datasets are edited by someone else, such as a domain expert.

Once you have your dataset on Confident AI, acces it by pulling it from the cloud:

from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.pull(alias="My first dataset")
print(dataset)

You'll often times want to process the pulled dataset before evaluating it, since test cases in a dataset are stored as Goldens, which might not always be ready for evaluation (ie. missing an actual_output). To see a concrete example and a more detailed explanation, visit the evaluating datasets section.

Running Evaluations
You can either run evaluations locally using deepeval, or on the cloud on a collection of metrics (which is also powered by deepeval). Most of the time, running evaluations locally is preferred because it allows for greater flexibility in metric customization. Using the previously pulled daataset, we can run an evaluation:

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
...

evaluate(dataset, metrics=[AnswerRelevancyMetric()])

You'll get a sharable testing report generated for you on Confident AI once your evaluation has completed. If you have more than two testing reports, you can also compare them to catch any regressions.

info
You can also log hyperparameters via the evaluate() function:

from deepeval import evaluate
...

evaluate(
    test_cases=[...],
    metrics=[...],
    hyperparameters={"model": "gpt4o", "prompt template": "..."}
)

Feel free to execute this in a nested for loop to figure out which combination gives the best results.

Monitoring LLM Outputs
Confident AI allows anyone to monitor, trace, and evaluate LLM outputs in real-time. A single API request is all that's required, and this would typically happen at your servers right before returning an LLM response to your users:

import openai
import deepeval

def sync_without_stream(user_message: str):
    model = "gpt-4-turbo"
    response = openai.ChatCompletion.create(
    model=model,
    messages=[{"role": "user", "content": user_message}]
    )
    output = response["choices"][0]["message"]["content"]

    # Run monitor() synchronously
    deepeval.monitor(input=user_message, output=output, model=model, event_name="RAG chatbot")
    return output

print(sync_without_stream("Tell me a joke."))

Collecting Human Feedback
Confident AI allows you to send human feedback on LLM responses monitored in production, all via one API call by using the previously returned response_id from deepeval.monitor():

import deepeval
...

deepeval.send_feedback(
    response_id=response_id,
    provider="user",
    rating=7,
    explanation="Although the response is accurate, I think the spacing makes it hard to read."
)

Confident AI allows you to keep track of either "user" feedback (ie. feedback provided by end users interacting with your LLM application), or "reviewer" feedback (ie. feedback provided by reviewers manually checking the quality of LLM responses in production).





Introduction
Quick Summary
Evaluation refers to the process of testing your LLM application outputs, and requires the following components:

Test cases
Metrics
Evaluation dataset
Here's a diagram of what an ideal evaluation workflow looks like using deepeval:


Your test cases will typically be in a single python file, and executing them will be as easy as running deepeval test run:

deepeval test run test_example.py

tip
Click here for an end-to-end tutorial on how to evaluate an LLM medical chatbot using deepeval.

Metrics
deepeval offers 14+ evaluation metrics, most of which are evaluated using LLMs (visit the metrics section to learn why).

from deepeval.metrics import AnswerRelevancyMetric

answer_relevancy_metric = AnswerRelevancyMetric()

You'll need to create a test case to run deepeval's metrics.

Test Cases
In deepeval, a test case allows you to use evaluation metrics you have defined to unit test LLM applications.

from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
  input="Who is the current president of the United States of America?",
  actual_output="Joe Biden",
  retrieval_context=["Joe Biden serves as the current president of America."]
)

In this example, input mimics an user interaction with a RAG-based LLM application, where actual_output is the output of your LLM application and retrieval_context is the retrieved nodes in your RAG pipeline. Creating a test case allows you to evaluate using deepeval's default metrics:

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

answer_relevancy_metric = AnswerRelevancyMetric()
test_case = LLMTestCase(
  input="Who is the current president of the United States of America?",
  actual_output="Joe Biden",
  retrieval_context=["Joe Biden serves as the current president of America."]
)

answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.score)

Datasets
Datasets in deepeval is a collection of test cases. It provides a centralized interface for you to evaluate a collection of test cases using one or multiple metrics.

from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric

answer_relevancy_metric = AnswerRelevancyMetric()
test_case = LLMTestCase(
  input="Who is the current president of the United States of America?",
  actual_output="Joe Biden",
  retrieval_context=["Joe Biden serves as the current president of America."]
)

dataset = EvaluationDataset(test_cases=[test_case])
dataset.evaluate([answer_relevancy_metric])

note
You don't need to create an evaluation dataset to evaluate individual test cases. Visit the test cases section to learn how to assert inidividual test cases.

Synthesizer
In deepeval, the Synthesizer allows you to generate synthetic datasets. This is especially helpful if you don't have production data or you don't have a golden dataset to evaluate with.

from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(document_paths=['example.txt', 'example.docx', 'example.pdf'])

dataset = EvaluationDataset(goldens=goldens)


info
deepeval's Synthesizer is highly customizable, and you can learn more about it here.

Evaluating With Pytest
caution
Although deepeval integrates with Pytest, we highly recommend you to AVOID executing LLMTestCases directly via the pytest command to avoid any unexpected errors.

deepeval allows you to run evaluations as if you're using Pytest via our Pytest integration. Simply create a test file:

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

dataset = EvaluationDataset(test_cases=[...])

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric()
    assert_test(test_case, [answer_relevancy_metric])

And run the test file in the CLI:

deepeval test run test_example.py

There are TWO mandatory and ONE optional parameter when calling the assert_test() function:

test_case: an LLMTestCase
metrics: a list of metrics of type BaseMetric
[Optional] run_async: a boolean which when set to True, enables concurrent evaluation of all metrics. Defaulted to True.
info
@pytest.mark.parametrize is a decorator offered by Pytest. It simply loops through your EvaluationDataset to evaluate each test case individually.

Parallelization
Evaluate each test case in parallel by providing a number to the -n flag to specify how many processes to use.

deepeval test run test_example.py -n 4

Cache
Provide the -c flag (with no arguments) to read from the local deepeval cache instead of re-evaluating test cases on the same metrics.

deepeval test run test_example.py -c

info
This is extremely useful if you're running large amounts of test cases. For example, lets say you're running 1000 test cases using deepeval test run, but you encounter an error on the 999th test case. The cache functionality would allow you to skip all the previously evaluated 999 test cases, and just evaluate the remaining one.

Ignore Errors
The -i flag (with no arguments) allows you to ignore errors for metrics executions during a test run. An example of where this is helpful is if you're using a custom LLM and often find it generating invalid JSONs that will stop the execution of the entire test run.

deepeval test run test_example.py -i

tip
You can combine differnet flags, such as the -i, -c, and -n flag to execute any uncached test cases in parallel while ignoring any errors along the way:

deepeval test run test_example.py -i -c -n 2

Verbose Mode
The -v flag (with no arguments) allows you to turn on verbose_mode for all metrics ran using deepeval test run. Not supplying the -v flag will default each metric's verbose_mode to its value at instantiation.

deepeval test run test_example.py -v

note
When a metric's verbose_mode is True, it prints the intermediate steps used to calculate said metric to the console during evaluation.

Skip Test Cases
The -s flag (with no arguments) allows you to skip metric executions where the test case has missing//insufficient parameters (such as retrieval_context) that is required for evaluation. An example of where this is helpful is if you're using a metric such as the ContextualPrecisionMetric but don't want to apply it when the retrieval_context is None.

deepeval test run test_example.py -s

Identifier
The -id flag followed by a string allows you to name test runs and better identify them on Confident AI. An example of where this is helpful is if you're running automated deployment pipelines, have deployment IDs, or just want a way to identify which test run is which for comparison purposes.

deepeval test run test_example.py -id "My Latest Test Run"

Display Mode
The -d flag followed by a string of "all", "passing", or "failing" allows you to display only certain test cases in the terminal. For example, you can display "failing" only if you only care about the failing test cases.

deepeval test run test_example.py -d "failing"

Repeats
Repeat each test case by providing a number to the -r flag to specify how many times to rerun each test case.

deepeval test run test_example.py -r 2

Hooks
deepeval's Pytest integration allosw you to run custom code at the end of each evaluation via the @deepeval.on_test_run_end decorator:

test_example.py
...

@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")

Evaluating Without Pytest
Alternately, you can use deepeval's evaluate function. This approach avoids the CLI (if you're in a notebook environment), and allows for parallel test execution as well.

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(test_cases=[...])
answer_relevancy_metric = AnswerRelevancyMetric()

evaluate(dataset, [answer_relevancy_metric])

There are TWO mandatory and THIRTEEN optional arguments when calling the evaluate() function:

test_cases: a list of LLMTestCases OR ConversationalTestCases, or an EvaluationDataset. You cannot evaluate LLMTestCase/MLLMTestCases and ConversationalTestCases in the same test run.
metrics: a list of metrics of type BaseMetric.
[Optional] hyperparameters: a dict of type dict[str, Union[str, int, float]]. You can log any arbitrary hyperparameter associated with this test run to pick the best hyperparameters for your LLM application on Confident AI.
[Optional] identifier: a string that allows you to better identify your test run on Confident AI.
[Optional] run_async: a boolean which when set to True, enables concurrent evaluation of test cases AND metrics. Defaulted to True.
[Optional] throttle_value: an integer that determines how long (in seconds) to throttle the evaluation of each test case. You can increase this value if your evaluation model is running into rate limit errors. Defaulted to 0.
[Optional] max_concurrent: an integer that determines the maximum number of test cases that can be ran in parallel at any point in time. You can decrease this value if your evaluation model is running into rate limit errors. Defaulted to 100.
[Optional] skip_on_missing_params: a boolean which when set to True, skips all metric executions for test cases with missing parameters. Defaulted to False.
[Optional] ignore_errors: a boolean which when set to True, ignores all exceptions raised during metrics execution for each test case. Defaulted to False.
[Optional] verbose_mode: a optional boolean which when IS NOT None, overrides each metric's verbose_mode value. Defaulted to None.
[Optional] write_cache: a boolean which when set to True, uses writes test run results to DISK. Defaulted to True.
[Optional] display: a str of either "all", "failing" or "passing", which allows you to selectively decide which type of test cases to display as the final result. Defaulted to "all".
[Optional] use_cache: a boolean which when set to True, uses cached test run results instead. Defaulted to False.
[Optional] show_indicator: a boolean which when set to True, shows the evaluation progress indicator for each individual metric. Defaulted to True.
[Optional] print_results: a boolean which when set to True, prints the result of each evaluation. Defaulted to True.
tip
You can also replace dataset with a list of test cases, as shown in the test cases section.





Test Cases
Quick Summary
A test case is a blueprint provided by deepeval to unit test LLM outputs. There are two types of test cases in deepeval: LLMTestCase and ConversationalTestCase.

caution
Throughout this documentation, you should assume the term 'test case' refers to an LLMTestCase instead of a ConversationalTestCase.

While a ConversationalTestCase is a list of turns represented by LLMTestCases, an LLMTestCase is the most prominent type of test case in deepeval and is based on seven parameters:

input
actual_output
[Optional] expected_output
[Optional] context
[Optional] retrieval_context
[Optional] tools_called
[Optional] expected_tools
Here's an example implementation of a test case:

from deepeval.test_case import LLMTestCase, ToolCall

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    expected_output="You're eligible for a 30 day refund at no extra cost.",
    actual_output="We offer a 30-day full refund at no extra cost.",
    context=["All customers are eligible for a 30 day full refund at no extra cost."],
    retrieval_context=["Only shoes can be refunded."],
    tools_called=[ToolCall(name="WebSearch")]
)

info
Since deepeval is an LLM evaluation framework, the input and actual_output are always mandatory. However, this does not mean they are necessarily used for evaluation, and you can also add additional parameters such as the tools_called for each LLMTestCase.

To get your own sharable testing report with deepeval, sign up to Confident AI, or run deepeval login in the CLI:

deepeval login

LLM Test Case
An LLMTestCase in deepeval can be used to unit test LLM application (which can just be an LLM itself) outputs, which includes use cases such as RAG and LLM agents. It contains the necessary information (tools_called for agents, retrieval_context for RAG, etc.) to evaluate your LLM application for a given input.

ok

Different metrics will require a different combination of LLMTestCase parameters, but they all require an input and actual_output - regardless of whether they are used for evaluation for not. For example, you won't need expected_output, context, tools_called, and expected_tools if you're just measuring answer relevancy, but if you're evaluating hallucination you'll have to provide context in order for deepeval to know what the ground truth is.

With the exception of conversational metrics, which are metrics to evaluate conversations instead of individual LLM responses, you can use any LLM evaluation metric deepeval offers to evaluate an LLMTestCase.

note
You cannot use conversational metrics to evaluate an LLMTestCase. Conveniently, most metrics in deepeval are non-conversational.

Keep reading to learn which parameters in an LLMTestCase are required to evaluate different aspects of an LLM applications - ranging from pure LLMs, RAG pipelines, and even LLM agents.

Input
The input mimics a user interacting with your LLM application. The input is the direct input to your prompt template, and so SHOULD NOT CONTAIN your prompt template.

from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="Why did the chicken cross the road?",
    # Replace this with your actual LLM application
    actual_output="Quite frankly, I don't want to know..."
)

tip
You should NOT include prompt templates as part of a test case because hyperparameters such as prompt templates are independent variables that you try to optimize for based on the metric scores you get from evaluation.

If you're logged into Confident AI, you can associate hyperparameters such as prompt templates with each test run to easily figure out which prompt template gives the best actual_outputs for a given input:

deepeval login

test_file.py
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_llm():
    test_case = LLMTestCase(input="...", actual_output="...")
    answer_relevancy_metric = AnswerRelevancyMetric()
    assert_test(test_case, [answer_relevancy_metric])

# You should aim to make these values dynamic
@deepeval.log_hyperparameters(model="gpt-4o", prompt_template="...")
def hyperparameters():
    # You can also return an empty dict {} if there's no additional parameters to log
    return {
        "temperature": 1,
        "chunk size": 500
    }

deepeval test run test_file.py

Actual Output
The actual_output is simply what your LLM application returns for a given input. This is what your users are going to interact with. Typically, you would import your LLM application (or parts of it) into your test file, and invoke it at runtime to get the actual output.

# A hypothetical LLM application example
import chatbot

input = "Why did the chicken cross the road?"

test_case = LLMTestCase(
    input=input,
    actual_output=chatbot.run(input)
)

note
You may also choose to evaluate with precomputed actual_outputs, instead of generating actual_outputs at evaluation time.

Expected Output
The expected_output is literally what you would want the ideal output to be. Note that this parameter is optional depending on the metric you want to evaluate.

The expected output doesn't have to exactly match the actual output in order for your test case to pass since deepeval uses a variety of methods to evaluate non-deterministic LLM outputs. We'll go into more details in the metrics section.

# A hypothetical LLM application example
import chatbot

input = "Why did the chicken cross the road?"

test_case = LLMTestCase(
    input=input,
    actual_output=chatbot.run(input),
    expected_output="To get to the other side!"
)

Context
The context is an optional parameter that represents additional data received by your LLM application as supplementary sources of golden truth. You can view it as the ideal segment of your knowledge base relevant to a specific input. Context allows your LLM to generate customized outputs that are outside the scope of the data it was trained on.

In RAG applications, contextual information is typically stored in your selected vector database, which is represented by retrieval_context in an LLMTestCase and is not to be confused with context. Conversely, for a fine-tuning use case, this data is usually found in training datasets used to fine-tune your model. Providing the appropriate contextual information when constructing your evaluation dataset is one of the most challenging part of evaluating LLMs, since data in your knowledge base can constantly be changing.

Unlike other parameters, a context accepts a list of strings.

# A hypothetical LLM application example
import chatbot

input = "Why did the chicken cross the road?"

test_case = LLMTestCase(
    input=input,
    actual_output=chatbot.run(input),
    expected_output="To get to the other side!",
    context=["The chicken wanted to cross the road."]
)

note
Often times people confuse expected_output with context since due to their similar level of factual accuracy. However, while both are (or should be) factually correct, expected_output also takes aspects like tone and linguistic patterns into account, whereas context is strictly factual.

Retrieval Context
The retrieval_context is an optional parameter that represents your RAG pipeline's retrieval results at runtime. By providing retrieval_context, you can determine how well your retriever is performing using context as a benchmark.

# A hypothetical LLM application example
import chatbot

input = "Why did the chicken cross the road?"

test_case = LLMTestCase(
    input=input,
    actual_output=chatbot.run(input),
    expected_output="To get to the other side!",
    context=["The chicken wanted to cross the road."],
    retrieval_context=["The chicken liked the other side of the road better"]
)

note
Remember, context is the ideal retrieval results for a given input and typically come from your evaluation dataset, whereas retrieval_context is your LLM application's actual retrieval results. So, while they might look similar at times, they are not the same.

Tools Called
The tools_called parameter is an optional parameter that represents the tools your LLM agent actually invoked during execution. By providing tools_called, you can evaluate how effectively your LLM agent utilized the tools available to it.

note
The tools_called parameter accepts a list of ToolCall objects.

class ToolCall(BaseModel):
    name: str
    description: Optional[str] = None
    reasoning: Optional[str] = None
    output: Optional[Any] = None
    input_parameters: Optional[Dict[str, Any]] = None

A ToolCall object accepts 1 mandatory and 4 optional parameters:

name: a string representing the name of the tool.
[Optional] description: a string describing the tool's purpose.
[Optional] reasoning: A string explaining the agent's reasoning to use the tool.
[Optional] output: The tool's output, which can be of any data type.
[Optional] input_parameters: A dictionary with string keys representing the input parameters (and respective values) passed into the tool function.
# A hypothetical LLM application example
import chatbot

test_case = LLMTestCase(
    input="Why did the chicken cross the road?",
    actual_output=chatbot.run(input),
    # Replace this with the tools that were actually used
    tools_called=[
        ToolCall(
            name="Calculator Tool"
            description="A tool that calculates mathematical equations or expressions.",
            input={"user_input": "2+3"}
            output=5
        ),
        ToolCall(
            name="WebSearch Tool"
            reasoning="Knowledge base does not detail why the chicken crossed the road."
            input={"search_query": "Why did the chicken crossed the road?"}
            output="Because it wanted to, duh."
        )
    ]
)

info
tools_called and expected_tools are LLM test case parameters that are utilized only in agentic evaluation metrics. These parameters allow you to assess the tool usage correctness of your LLM application and ensure that it meets the expected tool usage standards.

Expected Tools
The expected_tools parameter is an optional parameter that represents the tools that ideally should have been used to generate the output. By providing expected_tools, you can assess whether your LLM application used the tools you anticipated for optimal performance.

# A hypothetical LLM application example
import chatbot

input = "Why did the chicken cross the road?"

test_case = LLMTestCase(
    input=input,
    actual_output=chatbot.run(input),
    # Replace this with the tools that were actually used
    tools_called=[
        ToolCall(
            name="Calculator Tool"
            description="A tool that calculates mathematical equations or expressions.",
            input={"user_input": "2+3"}
            output=5
        ),
        ToolCall(
            name="WebSearch Tool"
            reasoning="Knowledge base does not detail why the chicken crossed the road."
            input={"search_query": "Why did the chicken crossed the road?"}
            output="Because it wanted to, duh."
        )
    ]
    expected_tools=[
        ToolCall(
            name="WebSearch Tool"
            reasoning="Knowledge base does not detail why the chicken crossed the road."
            input={"search_query": "Why did the chicken crossed the road?"}
            output="Because it needed to escape from the hungry humans."
        )
    ]
)

Conversational Test Case
A ConversationalTestCase in deepeval is simply a list of conversation turns represented by a list of LLMTestCases. While an LLMTestCase represents an individual LLM system interaction, a ConversationalTestCase encapsulates a series of LLMTestCases that make up an LLM-based conversation. This is particular useful if you're looking to for example evaluate a conversation between a user and an LLM-based chatbot.

While you cannot use a conversational metric on an LLMTestCase, a ConversationalTestCase can be evaluated using both non-conversational and conversational metrics.

from deepeval.test_case import LLMTestCase, ConversationalTestCase

llm_test_case = LLMTestCase(
    # Replace this with your user input
    input="Why did the chicken cross the road?",
    # Replace this with your actual LLM application
    actual_output="Quite frankly, I don't want to know..."
)

test_case = ConversationalTestCase(turns=[llm_test_case])

note
Similar to how the term 'test case' refers to an LLMTestCase if not explicitly specified, the term 'metrics' also refer to non-conversational metrics throughout deepeval.

Turns
The turns parameter is a list of LLMTestCases and is basically a list of messages/exchanges in a user-LLM conversation. Different conversational metrics will require different LLM test case parameters for evaluation, while regular LLM system metrics will take the last LLMTestCase in a turn to carry out evaluation.

from deepeval.test_case import LLMTestCase, ConversationalTestCase

test_case = ConversationalTestCase(turns=[LLMTestCase(...)])

Did you know?
You can apply both non-conversational and conversational metrics to a ConversationalTestCase. Conversational metrics evaluate the entire conversational as a whole, and non-conversational metrics (which are metrics used for individual LLMTestCases), when applied to a ConversationalTestCase, will evaluate the last turn in a ConversationalTestCase. This is because it is more useful to evaluate the last best LLM actual_output given the previous conversation context, instead of all individual turns in a ConversationalTestCase.

Chatbot Role
The chatbot_role parameter is an optional parameter that specifies what role the chatbot is supposed to play. This is currently only required for the RoleAdherenceMetric, where it is particularly useful for a role-playing evaluation use case.

from deepeval.test_case import LLMTestCase, ConversationalTestCase

test_case = ConversationalTestCase(
    chatbot_role="...",
    turns=[LLMTestCase(...)]
)

MLLM Test Case
An MLLMTestCase in deepeval is designed to unit test outputs from MLLM (Multimodal Large Language Model) applications. Unlike an LLMTestCase, which only handles textual parameters, an MLLMTestCase accepts both text and image inputs and outputs. This is particularly useful for evaluating tasks such as text-to-image generation or MLLM-driven image editing.

caution
You may only evaluate MLLMTestCases using multimodal metrics such as VIEScore.

from deepeval.test_case import MLLMTestCase, MLLMImage

mllm_test_case = MLLMTestCase(
    # Replace this with your user input
    input=["Change the color of the shoes to blue.", MLLMImage(url="./shoes.png", local=True)]
    # Replace this with your actual MLLM application
    actual_output=["The original image of red shoes now shows the shoes in blue.", MLLMImage(url="https://shoe-images.com/edited-shoes", local=False)]
)


Input
The input mimics a user interacting with your MLLM application. Like an LLMTestCase input, an MLLMTestCase input is the direct input to your prompt template, and so SHOULD NOT CONTAIN your prompt template.

from deepeval.test_case import MLLMTestCase, MLLMImage

mllm_test_case = MLLMTestCase(
    input=["Change the color of the shoes to blue.", MLLMImage(url="./shoes.png", local=True)]
)

info
The input parameter accepts a list of strings and MLLMImages, which is a class specific deepeval. The MLLMImage class accepts an image path and automatically sets the local attribute to true or false depending on whether the image is locally stored or hosted online. By default, local is set to false.

### Example:

```python
from deepeval.test_case import MLLMImage

# Example of using the MLLMImage class
image_input = MLLMImage(image_path="path/to/image.jpg")

# image_input.local will automatically be set to `true` if the image is local
# and `false` if the image is hosted online.

Actual Output
The actual_output is simply what your MLLM application returns for a given input. Similarly, it also accepts a list of strings and MLLMImages.

from deepeval.test_case import MLLMTestCase, MLLMImage

mllm_test_case = MLLMTestCase(
    input=["Change the color of the shoes to blue.", MLLMImage(url="./shoes.png", local=True)],
    actual_output=["The original image of red shoes now shows the shoes in blue.", MLLMImage(url="https://shoe-images.com/edited-shoes", local=False)]
)


Assert A Test Case
Before we begin going through the final sections, we highly recommend you to login to Confident AI (the platform powering deepeval) via the CLI. This way, you can keep track of all evaluation results generated each time you execute deepeval test run.

deepeval login

Similar to Pytest, deepeval allows you to assert any test case you create by calling the assert_test function by running deepeval test run via the CLI.

A test case passes only if all metrics passes. Depending on the metric, a combination of input, actual_output, expected_output, context, and retrieval_context is used to ascertain whether their criterion have been met.

test_assert_example.py
# A hypothetical LLM application example
import chatbot
import deepeval
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

def test_assert_example():
    input = "Why did the chicken cross the road?"
    test_case = LLMTestCase(
        input=input,
        actual_output=chatbot.run(input),
        context=["The chicken wanted to cross the road."],
    )
    metric = HallucinationMetric(threshold=0.7)
    assert_test(test_case, metrics=[metric])


# Optional. Log hyperparameters to pick the best hyperparameter for your LLM application
# using Confident AI. (run `deepeval login` in the CLI to login)
@deepeval.log_hyperparameters(model="gpt-4", prompt_template="...")
def hyperparameters():
    # Return a dict to log additional hyperparameters.
    # You can also return an empty dict {} if there's no additional parameters to log
    return {
        "temperature": 1,
        "chunk size": 500
    }

There are TWO mandatory and ONE optional parameter when calling the assert_test() function:

test_case: an LLMTestCase
metrics: a list of metrics of type BaseMetric
[Optional] run_async: a boolean which when set to True, enables concurrent evaluation of all metrics. Defaulted to True.
info
The run_async parameter overrides the async_mode property of all metrics being evaluated. The async_mode property, as you'll learn later in the metrics section, determines whether each metric can execute asynchronously.

To execute the test cases, run deepeval test run via the CLI, which uses deepeval's Pytest integration under the hood to execute these tests. You can also include an optional -n flag follow by a number (that determines the number of processes that will be used) to run tests in parallel.

deepeval test run test_assert_example.py -n 4

Evaluate Test Cases in Bulk
Lastly, deepeval offers an evaluate function to evaluate multiple test cases at once, which similar to assert_test but without the need for Pytest or the CLI.

# A hypothetical LLM application example
import chatbot
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input=input,
    actual_output=chatbot.run(input),
    context=["The chicken wanted to cross the road."],
)

metric = HallucinationMetric(threshold=0.7)
evaluate([test_case], [metric])

There are TWO mandatory and THIRTEEN optional arguments when calling the evaluate() function:

test_cases: a list of LLMTestCases OR ConversationalTestCases, or an EvaluationDataset. You cannot evaluate LLMTestCase/MLLMTestCases and ConversationalTestCases in the same test run.
metrics: a list of metrics of type BaseMetric.
[Optional] hyperparameters: a dict of type dict[str, Union[str, int, float]]. You can log any arbitrary hyperparameter associated with this test run to pick the best hyperparameters for your LLM application on Confident AI.
[Optional] identifier: a string that allows you to better identify your test run on Confident AI.
[Optional] run_async: a boolean which when set to True, enables concurrent evaluation of test cases AND metrics. Defaulted to True.
[Optional] throttle_value: an integer that determines how long (in seconds) to throttle the evaluation of each test case. You can increase this value if your evaluation model is running into rate limit errors. Defaulted to 0.
[Optional] max_concurrent: an integer that determines the maximum number of test cases that can be ran in parallel at any point in time. You can decrease this value if your evaluation model is running into rate limit errors. Defaulted to 100.
[Optional] skip_on_missing_params: a boolean which when set to True, skips all metric executions for test cases with missing parameters. Defaulted to False.
[Optional] ignore_errors: a boolean which when set to True, ignores all exceptions raised during metrics execution for each test case. Defaulted to False.
[Optional] verbose_mode: a optional boolean which when IS NOT None, overrides each metric's verbose_mode value. Defaulted to None.
[Optional] write_cache: a boolean which when set to True, uses writes test run results to DISK. Defaulted to True.
[Optional] display: a str of either "all", "failing" or "passing", which allows you to selectively decide which type of test cases to display as the final result. Defaulted to "all".
[Optional] use_cache: a boolean which when set to True, uses cached test run results instead. Defaulted to False.
[Optional] show_indicator: a boolean which when set to True, shows the evaluation progress indicator for each individual metric. Defaulted to True.
[Optional] print_results: a boolean which when set to True, prints the result of each evaluation. Defaulted to True.
DID YOU KNOW?
Similar to assert_test, evaluate allows you to log and view test results and the hyperparameters associated with each on Confident AI.

deepeval login

from deepeval import evaluate
...

evaluate(
    test_cases=[test_case],
    metrics=[metric],
    hyperparameters={"model": "gpt4o", "prompt template": "..."}
)

For more examples of evaluate, visit the datasets section.

Labeling Test Cases for Confident AI
If you're using Confident AI, the optional name parameter allows you to provide a string identifier to label LLMTestCases and ConversationalTestCases for you to easily search and filter for on Confident AI. This is particularly useful if you're importing test cases from an external datasource.

from deepeval.test_case import LLMTestCase, ConversationalTestCase

test_case = LLMTestCase(name="my-external-unique-id", ...)
convo_test_case = ConversationalTestCase(name="my-external-unique-id", ...)
