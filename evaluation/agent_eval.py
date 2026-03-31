from strands_evals import Case, Experiment
from strands_evals.evaluators import HelpfulnessEvaluator, GoalSuccessRateEvaluator, FaithfulnessEvaluator, ToolSelectionAccuracyEvaluator, OutputEvaluator, TrajectoryEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry
import sys
import os
import asyncio
import json
import boto3

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_secrets():
    secret_name = "dh-agent/config"
    region_name = "us-east-1"

    client = boto3.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    secrets = json.loads(response['SecretString'])

    for key, value in secrets.items():
        os.environ[key] = str(value)

load_secrets()

from src.orchestration.orchestrator import Orchestrator

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

async def get_response(case: Case) -> str:
    agent = Orchestrator()
    agent.agent.trace_attributes = {
        "gen_ai.conversation.id": case.session_id,
        "session.id": case.session_id
    }
    response = agent.ask(case.input)

    finished_spans = memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id=case.session_id)

    return {"output": str(response), "trajectory": session}

OUTPUT_RUBRIC = """
Evaluate the response based on:
1. Accuracy - Is the information correct?
2. Completeness - Does it fully answer the question?
3. Clarity - Is it easy to understand?

Score 1.0 if all criteria are met excellently.
Score 0.5 if some criteria are partially met.
Score 0.0 if the response is inadequate.
"""

TRAJECTORY_RUBRIC = """
The trajectory should be in the correct order with all of the steps as the expected.
The agent should know when and what action is logical. Strictly score 0 if any step is missing.
"""

evaluators = [
    OutputEvaluator(rubric=OUTPUT_RUBRIC, model='us.amazon.nova-pro-v1:0'),
    TrajectoryEvaluator(rubric=TRAJECTORY_RUBRIC, model='us.amazon.nova-pro-v1:0'),
    HelpfulnessEvaluator(model='us.amazon.nova-pro-v1:0'),
    FaithfulnessEvaluator(model='us.amazon.nova-pro-v1:0'),
    ToolSelectionAccuracyEvaluator(model='us.amazon.nova-pro-v1:0'),
    GoalSuccessRateEvaluator(model='us.amazon.nova-pro-v1:0')
]

EVAL_MODE = "general"
INPUT_FILE = f"datasets/eval_{EVAL_MODE}.json"
with open(INPUT_FILE, 'r') as f:
    EVAL_DATA = json.load(f)

test_cases = []

for conversation in EVAL_DATA:
    test_cases.append(Case[str, str](
            input=conversation["query"],
            expected_output=conversation["reference"],
            metadata=conversation["metadata"]
    ))

async def run_async_evaluation():
    experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)
    reports = await experiment.run_evaluations_async(get_response)

    for report in reports:
        report.run_display()

    return reports

if __name__ == "__main__":
    report = asyncio.run(run_async_evaluation())