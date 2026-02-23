import boto3
import os
import uuid
import time
import json
from langchain_aws import ChatBedrock
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness, 
    ToolCallAccuracy, 
    ToolCallF1, 
    TopicAdherenceScore, 
    AgentGoalAccuracyWithoutReference, 
    SemanticSimilarity
)
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample, EvaluationDataset
from ragas.integrations.amazon_bedrock import convert_to_ragas_messages
from ragas.messages import ToolCall
from ragas import evaluate
from dotenv import load_dotenv

load_dotenv()

client = boto3.client("bedrock-agent-runtime")
agent_id = os.getenv("AGENT_ID")
alias_id = os.getenv("ALIAS_ID")
bedrock_llm = ChatBedrock(model_id="us.amazon.nova-pro-v1:0", region_name="us-east-1")
evaluator_llm = LangchainLLMWrapper(bedrock_llm)

EVAL_MODE = "general"
INPUT_FILE = f"datasets/eval_{EVAL_MODE}.json"

def invokeAgent(query, session_id, session_state=dict()):
    agentResponse = client.invoke_agent(
        inputText=query,
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        enableTrace=True,
        endSession=False,
        sessionState=session_state,
    )
    event_stream = agentResponse["completion"]
    traces = []
    agent_answer = ""
    for event in event_stream:
        if "chunk" in event:
            agent_answer = event["chunk"]["bytes"].decode("utf8")
        elif "trace" in event:
            traces.append(event["trace"])
    return agent_answer, traces

with open(INPUT_FILE, 'r') as f:
    EVAL_DATA = json.load(f)

metrics_config = {
    "general": {
        "single": [FactualCorrectness(llm=evaluator_llm)],
        "multi": [ToolCallAccuracy()]
    },
    "aggregation": {
        "multi": [AgentGoalAccuracyWithoutReference(llm=evaluator_llm)]
    },
    # "followups": {
    #     "single": [TopicAdherenceScore(llm=evaluator_llm)],
    #     "multi": [ToolCallF1()]
    # },
    # "vqa": {
    #     "single": [SemanticSimilarity(llm=evaluator_llm)],
    #     "multi": [ToolCallAccuracy()]
    # },
    "outofscope": {
        "multi": [AgentGoalAccuracyWithoutReference(llm=evaluator_llm), ToolCallAccuracy()] 
    }
}

mode_metrics = metrics_config.get(EVAL_MODE)

single_samples = []
multi_samples = []

for item in EVAL_DATA:
    query = item["query"]
    reference = item["reference"]
    
    session_id = item.get("session_id", str(uuid.uuid4()))
    
    agent_answer, traces = invokeAgent(query, session_id)
    ragas_messages = convert_to_ragas_messages(traces)
    
    if mode_metrics["single"]:
        single_samples.append(SingleTurnSample(
            user_input=query,
            response=agent_answer,
            reference=reference
        ))

    ref_tool_calls = [
        ToolCall(name=tool["name"], args=tool["args"]) 
        for tool in item.get("expected_tools", [])
    ]

    if mode_metrics["multi"]:
        multi_samples.append(MultiTurnSample(
            user_input=ragas_messages,
            response=agent_answer,
            reference=reference,
            reference_tool_calls=ref_tool_calls
        ))
    
    print(f"[{EVAL_MODE}] Ran query: {query[:40]}...")
    time.sleep(2.5)

os.makedirs("results", exist_ok=True)

if mode_metrics["single"]:
    single_ds = EvaluationDataset(samples=single_samples)
    res_single = evaluate(dataset=single_ds, metrics=mode_metrics["single"])
    res_single.to_pandas().to_csv(f"results/{EVAL_MODE}_single_results.csv", index=False)

if mode_metrics["multi"]:
    multi_ds = EvaluationDataset(samples=multi_samples)
    res_multi = evaluate(dataset=multi_ds, metrics=mode_metrics["multi"])
    res_multi.to_pandas().to_csv(f"results/{EVAL_MODE}_multi_results.csv", index=False)

print(f"Done. Saved results for {EVAL_MODE}.")