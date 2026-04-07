from strands_evals import Case, Experiment, ActorSimulator
from strands_evals.evaluators import HelpfulnessEvaluator, GoalSuccessRateEvaluator, FaithfulnessEvaluator, ToolSelectionAccuracyEvaluator, OutputEvaluator, TrajectoryEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands.models import BedrockModel
from datetime import datetime
from pathlib import Path
import sys
import os
import json
import boto3

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

os.chdir(project_root)

def load_secrets():
    secret_name = "dh-agent/config"
    region_name = "us-east-1"

    client = boto3.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    secrets = json.loads(response['SecretString'])

    for key, value in secrets.items():
        os.environ[key] = str(value)

load_secrets()

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

OUTPUT_RUBRIC = """
Evaluate the response based on:
1. Accuracy - Is the information correct?
2. Completeness - Does it fully answer the question?
3. Clarity - Is it easy to understand?

Score 1.0 if all criteria are met excellently.
Score 0.5 if some criteria are partially met.
Score 0.0 if the response is inadequate.
"""

output_evaluator = OutputEvaluator(rubric=OUTPUT_RUBRIC, model=BedrockModel(model_id="us.amazon.nova-2-lite-v1:0", temperature=0.0))
helpfulness_evaluator = HelpfulnessEvaluator(model=BedrockModel(model_id="us.amazon.nova-2-lite-v1:0", temperature=0.0))
faithfulness_evaluator = FaithfulnessEvaluator(model=BedrockModel(model_id="us.amazon.nova-2-lite-v1:0", temperature=0.0))
tool_evaluator = ToolSelectionAccuracyEvaluator(model=BedrockModel(model_id="us.amazon.nova-2-lite-v1:0", temperature=0.0))
goal_evaluator = GoalSuccessRateEvaluator(model=BedrockModel(model_id="us.amazon.nova-2-lite-v1:0", temperature=0.0))

async def get_multiturn_response(case: Case) -> str:
    from src.orchestration.orchestrator import Orchestrator

    simulator = ActorSimulator.from_case_for_user_simulator(
        case=case,
        model='us.amazon.nova-pro-v1:0',
        max_turns=5
    )

    agent = Orchestrator()
    agent.agent.trace_attributes = {
        "gen_ai.conversation.id": case.session_id,
        "session.id": case.session_id
    }
    
    conversation_history = []
    all_spans = []
    user_message = case.input
    while simulator.has_next():
        memory_exporter.clear()

        agent_response = agent.ask(user_message)
        agent_message = str(agent_response)
        
        turn_spans = list(memory_exporter.get_finished_spans())
        all_spans.extend(turn_spans)

        conversation_history.append({
            "role": "agent",
            "message": agent_message
        })

        user_result = simulator.act(agent_message)
        user_message = str(user_result.structured_output.message)

        conversation_history.append({
            "role": "user",
            "message": user_message,
            "reasoning": user_result.structured_output.reasoning
        })

    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(all_spans, session_id=case.session_id)

    return {"output": agent_message, "trajectory": session, "conversation_history": conversation_history}

async def get_response(case: Case) -> str:
    from src.orchestration.orchestrator import Orchestrator
    
    memory_exporter.clear()
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

def create_dataset(mode="general"):
    input_data = f"evaluation/datasets/eval_{mode}.json"
    with open(input_data, 'r') as f:
        eval_data = json.load(f)

    test_cases = []

    for conversation in eval_data:
        test_cases.append(Case[str, str](
            input=conversation["query"],
            expected_output=conversation["reference"],
            metadata=conversation["metadata"],
            expected_trajectory=conversation["expected_trajectory"]
        ))
    
    return test_cases

async def run_async_evaluation(mode, test_cases, response_fn):
    evaluators = [output_evaluator, helpfulness_evaluator, faithfulness_evaluator, tool_evaluator, goal_evaluator]
    experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("evaluation/results") / mode / timestamp
    reports_dir = base_dir / "reports"
    
    reports_dir.mkdir(parents=True, exist_ok=True)

    reports = await experiment.run_evaluations_async(response_fn)

    experiment.to_file(base_dir / "experiment_config.json")

    for report in reports:
        print(f"\n{'='*60}")
        print(f"Evaluator: {report.evaluator_name}")
        print(f"{'='*60}")
        report.run_display()

        report_data = {
            "evaluator": report.evaluator_name,
            "overall_score": report.overall_score,
            "scores": report.scores,
            "test_passes": report.test_passes,
            "reasons": report.reasons,
            "timestamp": timestamp
        }

        report_file = reports_dir / f"{report.evaluator_name}_report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

    print(f"\n--- Evaluation Complete ---")
    print(f"Results saved to: {base_dir.absolute()}")

    return reports