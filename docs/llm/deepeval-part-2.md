Datasets
Quick Summary
In deepeval, an evaluation dataset, or just dataset, is a collection of LLMTestCases and/or Goldens. There are three approaches to evaluating datasets in deepeval:

using @pytest.mark.parametrize and assert_test
using evaluate
using confident_evaluate (evaluates on Confident AI instead of locally)
note
Evaluating a dataset means exactly the same as evaluating your LLM system, because by definition a dataset contains all the information produced by your LLM needed for evaluation.

You should also aim to group test cases of a certain category together in an EvaluationDataset. This will allow you to follow best practices:

Ensure telling test coverage: A well-structured dataset should reflect the full range of real-world inputs the LLM is expected to handle. This includes diverse linguistic styles, varying levels of complexity, and edge cases that challenge the model’s robustness. Don't just include test cases that are easy to pass.
Focused, quantitative test cases: A dataset should be designed with a clear evaluation scope while allowing for multiple relevant performance metrics. Rather than being overly broad or too narrow, it should be structured to provide meaningful, statistically reliable insights.
Define clear objectives: Each dataset should align with a well-defined evaluation goal, ensuring that test cases serve a specific analytical purpose. While organizing test cases into distinct datasets can provide clarity, unnecessary fragmentation should be avoided if multiple aspects of an LLM’s performance are best assessed together.
info
If you don't already have an EvaluationDataset, a great starting point is to simply write down the prompts you're currently using to manually eyeball your LLM outputs. You can also do this on Confident AI which integrates 100% with deepeval:

Full documentation for datasets on Confident AI [here.](/confident-ai/confident-ai-evaluation-dataset-management)
Create An Evaluation Dataset
An EvaluationDataset in deepeval is simply a collection of LLMTestCases and/or Goldens.

info
A Golden is extremely very similar to an LLMTestCase, but they are more flexible as they do not require an actual_output at initialization. On the flip side, whilst test cases are always ready for evaluation, a golden isn't.

With Test Cases
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

first_test_case = LLMTestCase(input="...", actual_output="...")
second_test_case = LLMTestCase(input="...", actual_output="...")

test_cases = [first_test_case, second_test_case]

dataset = EvaluationDataset(test_cases=test_cases)

You can also append a test case to an EvaluationDataset through the test_cases instance variable:

...

dataset.test_cases.append(test_case)
# or
dataset.add_test_case(test_case)

With Goldens
You should opt to initialize EvaluationDatasets with goldens if you're looking to generate LLM outputs at evaluation time. This usually means your original dataset does not contain precomputed outputs, but only the inputs you want to evaluate your LLM (application) on.

from deepeval.dataset import EvaluationDataset, Golden

first_golden = Golden(input="...")
second_golden = Golden(input="...")

dataset = EvaluationDataset(goldens=goldens)
print(dataset.goldens)

tip
A Golden and LLMTestCase contains almost an identical class signature, so technically you can also supply other parameters such as the actual_output when creating a Golden.

Generate An Evaluation Dataset
caution
We highly recommend you to checkout the Synthesizer page to see the customizations available and how data synthesization work in deepeval. All methods in an EvaluationDataset that can be used to generate goldens uses the Synthesizer under the hood and has exactly the same function signature as corresponding methods in the Synthesizer.

deepeval offers anyone the ability to easily generate synthetic datasets from documents locally on your machine. This is especially helpful if you don't have an evaluation dataset prepared beforehand.

from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.generate_goldens_from_docs(document_paths=['example.txt', 'example.docx', 'example.pdf'])

In this example, we've used the generate_goldens_from_docs method, which one one of the three generation methods offered by deepeval's Synthesizer. The three methods include:

generate_goldens_from_docs(): useful for generating goldens to evaluate your LLM application based on contexts extracted from your knowledge base in the form of documents.
generate_goldens_from_contexts(): useful for generating goldens to evaluate your LLM application based on a list of prepared context.
generate_goldens_from_scratch(): useful for generating goldens to evaluate your LLM application without relying on contexts from a knowledge base.
Under the hood, these 3 methods calls the corresponding methods in deepeval's Synthesizer with the exact same parameters, with an addition of a synthesizer parameter for you to customize your generation pipeline.

from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
...

# Use gpt-3.5-turbo instead
synthesizer = Synthesizer(model="gpt-3.5-turbo")
dataset.generate_goldens_from_docs(
    synthesizer=synthesizer,
    document_paths=['example.pdf'],
    max_goldens_per_document=2
)

info
deepeval's Synthesizer uses a series of evolution techniques to complicate and make generated goldens more realistic to human prepared data. For more information on how deepeval's Synthesizer works, visit the synthesizer section.

Load an Existing Dataset
deepeval offers support for loading datasetes stored in JSON files, CSV files, and hugging face datasets into an EvaluationDataset as either test cases or goldens.

From Confident AI
You can load entire datasets on Confident AI's cloud in one line of code.

from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.pull(alias="My Evals Dataset")

Did Your Know?
You can create, annotate, and comment on datasets on Confident AI? You can also upload datasets in CSV format, or push synthetic datasets created in deepeval to Confident AI in one line of code.

For more information, visit the Confident AI datasets section.

From JSON
You can loading an existing EvaluationDataset you might have generated elsewhere by supplying a file_path to your .json file as either test cases or goldens. Your .json file should contain an array of objects (or list of dictionaries).

from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()

# Add as test cases
dataset.add_test_cases_from_json_file(
    # file_path is the absolute path to you .json file
    file_path="example.json",
    input_key_name="query",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context",
    retrieval_context_key_name="retrieval_context",
)

# Or, add as goldens
dataset.add_goldens_from_json_file(
    # file_path is the absolute path to you .json file
    file_path="example.json",
    input_key_name="query"
)

info
Loading datasets as goldens are especially helpful if you're looking to generate LLM actual_outputs at evaluation time. You might find yourself in this situation if you are generating data for testing or using historical data from production.

From CSV
You can add test cases or goldens into your EvaluationDataset by supplying a file_path to your .csv file. Your .csv file should contain rows that can be mapped into LLMTestCases through their column names.

Remember, parameters such as context should be a list of strings and in the context of CSV files, it means you have to supply a context_col_delimiter argument to tell deepeval how to split your context cells into a list of strings.

from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()

# Add as test cases
dataset.add_test_cases_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path="example.csv",
    input_col_name="query",
    actual_output_col_name="actual_output",
    expected_output_col_name="expected_output",
    context_col_name="context",
    context_col_delimiter= ";",
    retrieval_context_col_name="retrieval_context",
    retrieval_context_col_delimiter= ";"
)

# Or, add as goldens
dataset.add_goldens_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path="example.csv",
    input_col_name="query"
)

note
Since expected_output, context, retrieval_context, tools_called, and expected_tools are optional parameters for an LLMTestCase, these fields are similarily optional parameters when adding test cases from an existing dataset.

Evaluate Your Dataset Using deepeval
tip
Before we begin, we highly recommend logging into Confident AI to keep track of all evaluation results created by deepeval on the cloud:

deepeval login

With Pytest
deepeval utilizes the @pytest.mark.parametrize decorator to loop through entire datasets.

test_bulk.py
import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset


dataset = EvaluationDataset(test_cases=[...])

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    hallucination_metric = HallucinationMetric(threshold=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [hallucination_metric, answer_relevancy_metric])


@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")

info
Iterating through an dataset object implicitly loops through the test cases in an dataset. To iterate through goldens, you can do it by accessing dataset.goldens instead.

To run several tests cases at once in parallel, use the optional -n flag followed by a number (that determines the number of processes that will be used) when executing deepeval test run:

deepeval test run test_bulk.py -n 3

Without Pytest
You can use deepeval's evaluate function to evaluate datasets. This approach avoids the CLI, but does not allow for parallel test execution.

from deepeval import evaluate
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(test_cases=[...])
hallucination_metric = HallucinationMetric(threshold=0.3)
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

dataset.evaluate([hallucination_metric, answer_relevancy_metric])

# You can also call the evaluate() function directly
evaluate(dataset, [hallucination_metric, answer_relevancy_metric])

info
Visit the test cases section to learn what argument the evaluate() function accepts.

Evaluate Your Dataset on Confident AI
Instead of running evaluations locally using your own evaluation LLMs via deepeval, you can choose to run evaluations on Confident AI's infrastructure instead. First, login to Confident AI:

deepeval login

Then, define metrics by creating an experiment on Confident AI. You can start running evaluations immediately by simply sending over your evaluation dataset and providing the name of the experiment you previously created via deepeval:

from deepeval import confident_evaluate
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(test_cases=[...])

confident_evaluate(experiment_name="My First Experiment", dataset)

tip
You can find the full tutorial on running evaluations on Confident AI here.









Introduction
Quick Summary
In deepeval, a metric serves as a standard of measurement for evaluating the performance of an LLM output based on a specific criteria of interest. Essentially, while the metric acts as the ruler, a test case represents the thing you're trying to measure. deepeval offers a range of default metrics for you to quickly get started with, such as:

G-Eval
DAG (Deep Acyclic Graph)
RAG:
Answer Relevancy
Faithfulness
Contextual Relevancy
Contextual Precision
Contextual Recall
Agents:
Tool Correctness
Task Completion
Others:
Json Correctness
Ragas
Hallucination
Toxicity
Bias
Summarization
deepeval also offers conversational metrics, which are metrics used to evaluate conversations instead of individual, granular LLM interactions. These include:

Conversational G-Eval
Knowledge Retention
Role Adherence
Conversation Completeness
Conversation Relevancy
You can also easily develop your own custom evaluation metrics in deepeval. All metrics are measured on a test case. Visit the test cases section to learn how to apply any metric on test cases for evaluation.

note
Your LLM application can be benchmarked by providing a list of metrics and test cases:

from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate

evaluate(test_cases=[...], metrics=[AnswerRelevancyMetric()])

You should also login to the deepeval cloud platform, Confident AI, before running evaluate():

deepeval login

When you run an evaluation using the evaluate() function or deepeval test run, you get testing reports on Confident AI, which allows you to:

Analyze metric score distributions, averages, and median scores.
Inspect metric results including reasoning, errors (if any), and verbose logs (for debugging your evaluation model's chain of thought).
Download test data as CSV or JSON (if you're running a conversational test run).
Create datasets from the test cases in a test run.
Create a public link to share with external stakeholders that might be interested in your LLM evaluation process.
More information can be found on the Confident AI quickstart.

Types of Metrics
deepeval offers a wide range of custom and default metrics and all of them uses LLM-as-a-judge. There are two types of custom metrics, with varying degree of deterministicity:

G-Eval
DAG
The DAG metric is a decision-tree based LLM-evaluated metric, and is currently the most versitile metric deepeval has to offer. However, G-Eval is also extremely competent and takes no effort at all to setup so we recommend everyone to start with G-Eval and move to DAG if there's a need for it.

note
You can also inhert a BaseMetric class to create your own custom metric. They are extremely easy to create and almost 10% of all metrics ran using deepeval are self-built metrics.

deepeval also offers default metrics, which are pre-built for different LLM systems/use cases. For example, deepeval offers the famous RAG metrics out-of-the-box:

Answer Relevancy
Faithfulness
Contextual Relevancy
Contextual Precision
Contextual Recall
info
deepeval deliberately uses LLM-as-a-judge for all metrics because our experience tells us that they better align with human expectations when compared to traditional model based approaches.

deepeval's metrics are a step up to other implementations because they:

Make deterministic metric scores possible (when using DAGMetric).
Easily customizable (GEval and DAGMetric).
Are extra reliable as LLMs are only used for extremely confined tasks during evaluation to greatly reduce stochasticity and flakiness in scores.
Provide a comprehensive reason for the scores computed.
Can be computed using any LLM.
Can be customized by overriding evaluation prompts.
Integrated 100% with Confident AI.
All of deepeval's metrics output a score between 0-1. A metric is only successful if the evaluation score is equal to or greater than threshold, which is defaulted to 0.5 for all metrics.

If you're not sure which metric to use, join our discord community or run the follow command to find out:

deepeval recommend metrics

tip
All LLMs from OpenAI are available for LLM-Evals (metrics that use LLMs for evaluation). You can switch between models by providing a string corresponding to OpenAI's model names via the optional model argument when instantiating an LLM-Eval.

What About Non LLM-Evals?
If you're looking to use something like ROUGE, BLEU, or BLURT, etc. you can create a custom metric and use the scorer module available in deepeval for scoring by following this guide.

The scorer module is available but not documented because our experience tells us these scorers are not useful as LLM metrics where outputs require a high level of reasoning to evaluate.

Using OpenAI
To use OpenAI for deepeval's LLM-Evals (metrics evaluated using an LLM), supply your OPENAI_API_KEY in the CLI:

export OPENAI_API_KEY=<your-openai-api-key>

Alternatively, if you're working in a notebook enviornment (Jupyter or Colab), set your OPENAI_API_KEY in a cell:

 %env OPENAI_API_KEY=<your-openai-api-key>

note
Please do not include quotation marks when setting your OPENAI_API_KEY if you're working in a notebook enviornment.

Azure OpenAI
deepeval also allows you to use Azure OpenAI for metrics that are evaluated using an LLM. Run the following command in the CLI to configure your deepeval enviornment to use Azure OpenAI for all LLM-based metrics.

deepeval set-azure-openai --openai-endpoint=<endpoint> \
    --openai-api-key=<api_key> \
    --deployment-name=<deployment_name> \
    --openai-api-version=<openai_api_version> \
    --model-version=<model_version>

Note that the model-version is optional. If you ever wish to stop using Azure OpenAI and move back to regular OpenAI, simply run:

deepeval unset-azure-openai

Using Ollama
note
Before getting started, make sure your Ollama model is installed and running. You can also see the full list of available models by clicking on the previous link.

ollama run deepseek-r1:1.5b

To use Ollama models for your metrics, run deepeval set-ollama <model> in your CLI. For example:

deepeval set-ollama deepseek-r1:1.5b

Optionally, you can specify the base URL of your local Ollama model instance if you've defined a custom port. The default base URL is set to http://localhost:11434.

deepeval set-ollama deepseek-r1:1.5b \
    --base-url="http://localhost:11434"

To stop using your local Ollama model and move back to OpenAI, run:

deepeval unset-ollama

caution
The deepeval set-ollama command is used exclusively to configure LLM models. If you intend to use a custom embedding model from Ollama with the synthesizer, please refer to this section of the guide.

Other Local Providers
In additional to Ollama, deepeval also supports local LLM providers that offer an OpenAI API compatible endpoint like LM Studio. To use them with deepeval you need to configure them using the CLI. This will make deepeval use the local LLM model for all LLM-based metrics.

To configure any of those providers, you need to supply the base URL where the service is running. These are some of the most popular alternatives for base URLs:

LM Studio: http://localhost:1234/v1/
vLLM: http://localhost:8000/v1/
So, to configure a model using LM studio, use the following command:

deepeval set-local-model --model-name=<model_name> \
    --base-url="http://localhost:1234/v1/" \
    --api-key=<api-key>

note
For additional instructions about model availability and base URLs, consult each provider's documentation.

If you ever wish to stop using your local LLM model and move back to regular OpenAI, simply run:

deepeval unset-local-model

Using Any Custom LLM
deepeval allows you to use ANY custom LLM for evaluation. This includes LLMs from langchain's chat_model module, Hugging Face's transformers library, or even LLMs in GGML format.

This includes any of your favorite models such as:

Azure OpenAI
Claude via AWS Bedrock
Google Vertex AI
Mistral 7B
All the examples can be found here.

caution
We CANNOT guarantee that evaluations will work as expected when using a custom model. This is because evaluation requires high levels of reasoning and the ability to follow instructions such as outputing responses in valid JSON formats. To better enable custom LLMs output valid JSONs, read this guide.

Alternatively, if you find yourself running into JSON errors and would like to ignore it, use the -c and -i flag during deepeval test run:

deepeval test run test_example.py -i -c

The -i flag ignores errors while the -c flag utilizes the local deepeval cache, so for a partially successful test run you don't have to rerun test cases that didn't error.

Running Evaluations With Metrics
To run evaluations using any metric of your choice, simply provide a list of test cases to evaluate your metrics against:

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_case = LLMTestCase(input="...", actual_output="...")

evaluate(test_cases=[test_case], metrics=[AnswerRelevancyMetric()])

The evaluate() function or deepeval test run is the best way to run evaluations. They offer tons of features out of the box, including caching, parallelization, cost tracking, error handling, and integration with Confident AI.

tip
deepeval test run is deepeval's native Pytest integration, which allows you to run evals in CI/CD pipelines.

Measuring A Metric
You can also execute each metric individually. All metrics in deepeval, including custom metrics that you create:

can be executed via the metric.measure() method
can have its score accessed via metric.score, which ranges from 0 - 1
can have its score reason accessed via metric.reason
can have its status accessed via metric.is_successful()
can be used to evaluate test cases or entire datasets, with or without Pytest
has a threshold that acts as the threshold for success. metric.is_successful() is only true if metric.score is above/below threshold
has a strict_mode property, which when turned on enforces metric.score to a binary one
has a verbose_mode property, which when turned on prints metric logs whenever a metric is executed
In additional, all metrics in deepeval execute asynchronously by default. This behavior is something you can configure via the async_mode parameter when instantiating a metric.

tip
Visit an individual metric page to learn how they are calculated, and what is required when creating an LLMTestCase in order to execute it.

Here's a quick example.

export OPENAI_API_KEY=<your-openai-api-key>

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Initialize a test case
test_case = LLMTestCase(
    input="...",
    actual_output="...",
    retrieval_context=["..."]
)

# Initialize metric with threshold
metric = AnswerRelevancyMetric(threshold=0.5)

Using this metric, you can either execute it directly as a standalone to get its score and reason:

...

metric.measure(test_case)
print(metric.score)
print(metric.reason)

Or you can either assert a test case using assert_test() via deepeval test run:

test_file.py
from deepeval import assert_test
...

def test_answer_relevancy():
    assert_test(test_case, [metric])

deepeval test run test_file.py

Or using the evaluate function:

from deepeval import evaluate
...

evaluate([test_case], [metric])

Measuring Metrics in Async
When a metric's async_mode=True (which is the default value for all metrics), invocations of metric.measure() will execute its internal algorithms concurrently. However, it's important to note that while operations INSIDE measure() executes concurrently, the metric.measure() call itself still blocks the main thread.

info
Let's take the FaithfulnessMetric algorithm for example:

Extract all factual claims made in the actual_output
Extract all factual truths found in the retrieval_context
Compare extracted claims and truths to generate a final score and reason.
from deepeval.metrics import FaithfulnessMetric
...

metric = FaithfulnessMetric(async_mode=True)
metric.measure(test_case)
print("Metric finished!")

When async_mode=True, steps 1 and 2 executes concurrently (ie. at the same time) since they are independent of each other, while async_mode=False will cause steps 1 and 2 to execute sequentially instead (ie. one after the other).

In both cases, "Metric finished!" will wait for metric.measure() to finish running before printing, but setting async_mode to True would make the print statement appear earlier, as async_mode=True allows metric.measure() to run faster.

To measure multiple metrics at once and NOT block the main thread, use the asynchronous a_measure() method instead.

import asyncio
...

# Remember to use async
async def long_running_function():
    # These will all run at the same time
    await asyncio.gather(
        metric1.a_measure(test_case),
        metric2.a_measure(test_case),
        metric3.a_measure(test_case),
        metric4.a_measure(test_case)
    )
    print("Metrics finished!")

asyncio.run(long_running_function())

Debugging A Metric
You can turn on verbose_mode for ANY deepeval metric at metric initialization to debug a metric whenever the measure() or a_measure() method is called:

...

metric = AnswerRelevancyMetric(verbose_mode=True)
metric.measure(test_case)

note
Turning verbose_mode on will print the inner workings of a metric whenever measure() or a_measure() is called.

Customizing Metric Prompts
All of deepeval's metrics uses LLM-as-a-judge which comes with a set of default prompt templates unique to each metric that are used for evaluation. While deepeval has a laid out algorithm to each metric, you can still customize these prompt templates to improve the accuracy and stability of your evaluation scores. This can be done by providing a custom template class as the evaluation_template to your metric of choice (example below).

info
For example, in the AnswerRelevancyMetric, you might disagree with what we consider something to be "relevant", but with this capability you can now override any opinions deepeval has in its default evaluation prompts.

You're most likely to find this valuable if you're using a custom LLM, because deepeval's metrics are mostly adopted for OpenAI's models, which are in general more powerful than your choice of custom LLM.

note
This means you can better handle invalid JSON outputs (along with JSON confinement) which comes with weaker models, and provide better examples for in-context learning for your custom LLM judges for better metric accuracy.

Here's a quick example of how you can define a custom AnswerRelevancyTemplate and inject it into the AnswerRelevancyMetric through the evaluation_params parameter:

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

# Define custom template
class CustomTemplate(AnswerRelevancyTemplate):
    @staticmethod
    def generate_statements(actual_output: str):
        return f"""Given the text, breakdown and generate a list of statements presented.

Example:
Our new laptop model features a high-resolution Retina display for crystal-clear visuals.

{{
    "statements": [
        "The new laptop model has a high-resolution Retina display."
    ]
}}
===== END OF EXAMPLE ======

Text:
{actual_output}

JSON:
"""

# Inject custom template to metric
metric = AnswerRelevancyMetric(evaluation_template=CustomTemplate)
metric.measure(...)

tip
You can find examples of how this can be done in more detail on the Customize Your Template section of each individual metric page, which shows code examples, and a link to deepeval's GitHub showing the default templates currently used.

Edit this page












G-Eval
G-Eval is a framework that uses LLM-as-a-judge with chain-of-thoughts (CoT) to evaluate LLM outputs based on ANY custom criteria. The G-Eval metric is the most versatile type of metric deepeval has to offer, and is capable of evaluating almost any use case with human-like accuracy.

tip
Usually, a GEval metric will be used alongside one of the other metrics that are more system specific (such as ContextualRelevancyMetric for RAG, and TaskCompletionMetric for agents).

If you want custom but extremely deterministic metric scores, you can checkout deepeval's DAGMetric instead. It is also a custom metric, but allows you to run evaluations by constructing a LLM-powered decision trees.

Required Arguments
To use the GEval, you'll have to provide the following arguments when creating an LLMTestCase:

input
actual_output
You'll also need to supply any additional arguments such as expected_output and context if your evaluation criteria depends on these parameters.

Example
To create a custom metric that uses LLMs for evaluation, simply instantiate an GEval class and define an evaluation criteria in everyday language:

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)


There are THREE mandatory and SIX optional parameters required when instantiating an GEval class:

name: name of metric
criteria: a description outlining the specific evaluation aspects for each test case.
evaluation_params: a list of type LLMTestCaseParams. Include only the parameters that are relevant for evaluation.
[Optional] evaluation_steps: a list of strings outlining the exact steps the LLM should take for evaluation. If evaluation_steps is not provided, GEval will generate a series of evaluation_steps on your behalf based on the provided criteria. You can only provide either evaluation_steps OR criteria, and not both.
[Optional] threshold: the passing threshold, defaulted to 0.5.
[Optional] model: a string specifying which of OpenAI's GPT models to use, OR any custom LLM model of type DeepEvalBaseLLM. Defaulted to 'gpt-4o'.
[Optional] strict_mode: a boolean which when set to True, enforces a binary metric score: 1 for perfection, 0 otherwise. It also overrides the current threshold and sets it to 1. Defaulted to False.
[Optional] async_mode: a boolean which when set to True, enables concurrent execution within the measure() method. Defaulted to True.
[Optional] verbose_mode: a boolean which when set to True, prints the intermediate steps used to calculate said metric to the console, as outlined in the How Is It Calculated section. Defaulted to False.
danger
For accurate and valid results, only the parameters that are mentioned in criteria/evaluation_params should be included as a member of evaluation_params.

As mentioned in the metrics introduction section, all of deepeval's metrics return a score ranging from 0 - 1, and a metric is only successful if the evaluation score is equal to or greater than threshold, and GEval is no exception. You can access the score and reason for each individual GEval metric:

from deepeval.test_case import LLMTestCase
...

test_case = LLMTestCase(
    input="The dog chased the cat up the tree, who ran up the tree?",
    actual_output="It depends, some might consider the cat, while others might argue the dog.",
    expected_output="The cat."
)

# To run metric as a standalone
# correctness_metric.measure(test_case)
# print(correctness_metric.score, correctness_metric.reason)

evaluate(test_cases=[test_case], metrics=[correctness_metric])

As a standalone
You can also run GEval on a single test case as a standalone, one-off execution.

...

correctness_metric.measure(test_case)
print(correctness_metric.score, correctness_metric.reason)

caution
This is great for debugging or if you wish to build your own evaluation pipeline, but you will NOT get the benefits (testing reports, Confident AI platform) and all the optimizations (speed, caching, computation) the evaluate() function or deepeval test run offers.

What is G-Eval?
G-Eval is a framework originally from the paper “NLG Evaluation using GPT-4 with Better Human Alignment” that uses LLMs to evaluate LLM outputs (aka. LLM-Evals), and is one the best ways to create task-specific metrics.

The G-Eval algorithm first generates a series of evaluation steps for chain of thoughts (CoTs) prompting before using the generated steps to determine the final score via a "form-filling paradigm" (which is just a fancy way of saying G-Eval requires different LLMTestCase parameters for evaluation depending on the generated steps).

ok

After generating a series of evaluation steps, G-Eval will:

Create prompt by concatenating the evaluation steps with all the paramters in an LLMTestCase that is supplied to evaluation_params.
At the end of the prompt, ask it to generate a score between 1–5, where 5 is better than 1.
Take the probabilities of the output tokens from the LLM to normalize the score and take their weighted summation as the final result.
info
We highly recommend everyone to read this article on LLM evaluation metrics. It's written by the founder of deepeval and explains the rationale and algorithms behind the deepeval metrics, including GEval.

Here are the results from the paper, which shows how G-Eval outperforms all traditional, non-LLM evals that were mentioned earlier in this article:

ok

note
Although GEval is great it many ways as a custom, task-specific metric, it is NOT deterministic. If you're looking for more fine-grained, deterministic control over your metric scores, you should be using the DAGMetric instead.

How Is It Calculated?
Since G-Eval is a two-step algorithm that generates chain of thoughts (CoTs) for better evaluation, in deepeval this means first generating a series of evaluation_steps using CoT based on the given criteria, before using the generated steps to determine the final score using the parameters presented in an LLMTestCase.

When you provide evaluation_steps, the GEval metric skips the first step and uses the provided steps to determine the final score instead, make it more reliable across different runs. If you don't have a clear evaluation_stepss, what we've found useful is to first write a criteria which can be extremely short, and use the evaluation_steps generated by GEval for subsequent evaluation and fine-tuning of criteria.

Did Your Know?
In the original G-Eval paper, the authors used the the probabilities of the LLM output tokens to normalize the score by calculating a weighted summation.

This step was introduced in the paper because it minimizes bias in LLM scoring. This normalization step is automatically handled by deepeval by default (unless you're using a custom model).

Edit this page






Custom Metrics
note
This page is identical to the guide on building custom metrics which can be found here.

In deepeval, anyone can easily build their own custom LLM evaluation metric that is automatically integrated within deepeval's ecosystem, which includes:

Running your custom metric in CI/CD pipelines.
Taking advantage of deepeval's capabilities such as metric caching and multi-processing.
Have custom metric results automatically sent to Confident AI.
Here are a few reasons why you might want to build your own LLM evaluation metric:

You want greater control over the evaluation criteria used (and you think GEval or DAG is insufficient).
You don't want to use an LLM for evaluation (since all metrics in deepeval are powered by LLMs).
You wish to combine several deepeval metrics (eg., it makes a lot of sense to have a metric that checks for both answer relevancy and faithfulness).
info
There are many ways one can implement an LLM evaluation metric. Here is a great article on everything you need to know about scoring LLM evaluation metrics.

Rules To Follow When Creating A Custom Metric
1. Inherit the BaseMetric class
To begin, create a class that inherits from deepeval's BaseMetric class:

from deepeval.metrics import BaseMetric

class CustomMetric(BaseMetric):
    ...

This is important because the BaseMetric class will help deepeval acknowledge your custom metric during evaluation.

2. Implement the __init__() method
The BaseMetric class gives your custom metric a few properties that you can configure and be displayed post-evaluation, either locally or on Confident AI.

An example is the threshold property, which determines whether the LLMTestCase being evaluated has passed or not. Although the threshold property is all you need to make a custom metric functional, here are some additional properties for those who want even more customizability:

evaluation_model: a str specifying the name of the evaluation model used.
include_reason: a bool specifying whether to include a reason alongside the metric score. This won't be needed if you don't plan on using an LLM for evaluation.
strict_mode: a bool specifying whether to pass the metric only if there is a perfect score.
async_mode: a bool specifying whether to execute the metric asynchronously.
tip
Don't read too much into the advanced properties for now, we'll go over how they can be useful in later sections of this guide.

The __init__() method is a great place to set these properties:

from deepeval.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        # Optional
        evaluation_model: str,
        include_reason: bool = True,
        strict_mode: bool = True,
        async_mode: bool = True
    ):
        self.threshold = threshold
        # Optional
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode

3. Implement the measure() and a_measure() methods
The measure() and a_measure() method is where all the evaluation happens. In deepeval, evaluation is the process of applying a metric to an LLMTestCase to generate a score and optionally a reason for the score (if you're using an LLM) based on the scoring algorithm.

The a_measure() method is simply the asynchronous implementation of the measure() method, and so they should both use the same scoring algorithm.

info
The a_measure() method allows deepeval to run your custom metric asynchronously. Take the assert_test function for example:

from deepeval import assert_test

def test_multiple_metrics():
    ...
    assert_test(test_case, [metric1, metric2], run_async=True)

When you run assert_test() with run_async=True (which is the default behavior), deepeval calls the a_measure() method which allows all metrics to run concurrently in a non-blocking way.

Both measure() and a_measure() MUST:

accept an LLMTestCase as argument
set self.score
set self.success
You can also optionally set self.reason in the measure methods (if you're using an LLM for evaluation), or wrap everything in a try block to catch any exceptions and set it to self.error. Here's a hypothetical example:

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    ...

    def measure(self, test_case: LLMTestCase) -> float:
        # Although not required, we recommend catching errors
        # in a try block
        try:
            self.score = generate_hypothetical_score(test_case)
            if self.include_reason:
                self.reason = generate_hypothetical_reason(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            # set metric error and re-raise it
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        # Although not required, we recommend catching errors
        # in a try block
        try:
            self.score = await async_generate_hypothetical_score(test_case)
            if self.include_reason:
                self.reason = await async_generate_hypothetical_reason(test_case)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            # set metric error and re-raise it
            self.error = str(e)
            raise

tip
Often times, the blocking part of an LLM evaluation metric stems from the API calls made to your LLM provider (such as OpenAI's API endpoints), and so ultimately you'll have to ensure that LLM inference can indeed be made asynchronous.

If you've explored all your options and realize there is no asynchronous implementation of your LLM call (eg., if you're using an open-source model from Hugging Face's transformers library), simply reuse the measure method in a_measure():

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    ...

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

You can also click here to find an example of offloading LLM inference to a separate thread as a workaround, although it might not work for all use cases.

4. Implement the is_successful() method
Under the hood, deepeval calls the is_successful() method to determine the status of your metric for a given LLMTestCase. We recommend copy and pasting the code below directly as your is_successful() implementation:

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    ...

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

5. Name Your Custom Metric
Probably the easiest step, all that's left is to name your custom metric:

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CustomMetric(BaseMetric):
    ...

    @property
    def __name__(self):
        return "My Custom Metric"

Congratulations 🎉! You've just learnt how to build a custom metric that is 100% integrated with deepeval's ecosystem. In the following section, we'll go through a few real-life examples.

Building a Custom Non-LLM Eval
An LLM-Eval is an LLM evaluation metric that is scored using an LLM, and so a non-LLM eval is simply a metric that is not scored using an LLM. In this example, we'll demonstrate how to use the rouge score instead:

from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1"
        )
        self.success = self.score >= self.threshold
        return self.score

    # Async implementation of measure(). If async version for
    # scoring method does not exist, just reuse the measure method.
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"

note
Although you're free to implement your own rouge scorer, you'll notice that while not documented, deepeval additionally offers a scorer module for more traditional NLP scoring method and can be found here.

Be sure to run pip install rouge-score if rouge-score is not already installed in your environment.

You can now run this custom metric as a standalone in a few lines of code:

...

#####################
### Example Usage ###
#####################
test_case = LLMTestCase(input="...", actual_output="...", expected_output="...")
metric = RougeMetric()

metric.measure(test_case)
print(metric.is_successful())

Building a Custom Composite Metric
In this example, we'll be combining two default deepeval metrics as our custom metric, hence why we're calling it a "composite" metric.

We'll be combining the AnswerRelevancyMetric and FaithfulnessMetric, since we rarely see a user that cares about one but not the other.

from deepeval.metrics import BaseMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

class FaithfulRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: Optional[str] = "gpt-4-turbo",
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    def measure(self, test_case: LLMTestCase):
        try:
            relevancy_metric, faithfulness_metric = initialize_metrics()
            # Remember, deepeval's default metrics follow the same pattern as your custom metric!
            relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)

            # Custom logic to set score, reason, and success
            set_score_reason_success(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            # Set and re-raise error
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase):
        try:
            relevancy_metric, faithfulness_metric = initialize_metrics()
            # Here, we use the a_measure() method instead so both metrics can run concurrently
            await relevancy_metric.a_measure(test_case)
            await faithfulness_metric.a_measure(test_case)

            # Custom logic to set score, reason, and success
            set_score_reason_success(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            # Set and re-raise error
            self.error = str(e)
            raise

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success

    @property
    def __name__(self):
        return "Composite Relevancy Faithfulness Metric"


    ######################
    ### Helper methods ###
    ######################
    def initialize_metrics(self):
        relevancy_metric = AnswerRelevancyMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode
        )
        faithfulness_metric = FaithfulnessMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode
        )
        return relevancy_metric, faithfulness_metric

    def set_score_reason_success(
        self,
        relevancy_metric: BaseMetric,
        faithfulness_metric: BaseMetric
    ):
        # Get scores and reasons for both
        relevancy_score = relevancy_metric.score
        relevancy_reason = relevancy_metric.reason
        faithfulness_score = faithfulness_metric.score
        faithfulness_reason = faithfulness_reason.reason

        # Custom logic to set score
        composite_score = min(relevancy_score, faithfulness_score)
        self.score = 0 if self.strict_mode and composite_score < self.threshold else composite_score

        # Custom logic to set reason
        if include_reason:
            self.reason = relevancy_reason + "\n" + faithfulness_reason

        # Custom logic to set success
        self.success = self.score >= self.threshold

Now go ahead and try to use it:

test_llm.py
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
...

def test_llm():
    metric = FaithfulRelevancyMetric()
    test_case = LLMTestCase(...)
    assert_test(test_case, [metric])

deepeval test run test_llm.py

Edit this page














