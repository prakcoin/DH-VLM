from strands import Agent
from strands_evals import Case, Experiment, ActorSimulator
from strands_evals.evaluators import HelpfulnessEvaluator, GoalSuccessRateEvaluator, FaithfulnessEvaluator, ToolSelectionAccuracyEvaluator 
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestration.orchestrator import Orchestrator

telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

def get_response(case: Case) -> str:
    agent = Orchestrator()
    agent.agent.trace_attributes = {
        "gen_ai.conversation.id": case.session_id,
        "session.id": case.session_id
    },
    response = agent.ask(case.input)

    finished_spans = memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id=case.session_id)

    return {"output": str(response), "trajectory": session}

evaluators = [
    HelpfulnessEvaluator(),
    FaithfulnessEvaluator(),
]

test_cases = [
    Case[str, str](
        input="What does look 1 consist of?",
        expected_output="Look 1 consists of a 1B blazer, a leather tie, a pinstripe shirt, trousers, suede moto boots, aviator sunglasses, a leather belt, and a bandana bracelet.",
    ),
    # Case[str, str](
    #     input="What is the reference code for the beetle leather jacket?",
    #     expected_output="The reference code for the beetle leather jacket is 4HH5041101.",
    # ),
    # Case[str, str](
    #     input="What material is the jacket in look 2 made from?",
    #     expected_output="The jacket in look 2 is made from leather, more specifically calfskin.",
    # )
]

experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)
reports = experiment.run_evaluations(get_response)

print("=== Basic Output Evaluation Results ===")
reports[0].run_display()