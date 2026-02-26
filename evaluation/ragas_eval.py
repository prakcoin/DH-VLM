import boto3
import os
import uuid
import json
import pandas as pd
from langchain_aws import ChatBedrock, BedrockEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    FactualCorrectness, 
    SemanticSimilarity
)
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas import evaluate
from dotenv import load_dotenv

load_dotenv()

client = boto3.client("bedrock-agent-runtime")
agent_id = os.getenv("AGENT_ID")
alias_id = os.getenv("ALIAS_ID")
bedrock_llm = ChatBedrock(model_id="us.amazon.nova-pro-v1:0", region_name="us-east-1")
evaluator_llm = LangchainLLMWrapper(bedrock_llm)

bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.nova-2-multimodal-embeddings-v1:0",
    region_name="us-east-1"
)

embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)

EVAL_MODE = "outofscope"
INPUT_FILE = f"datasets/eval_{EVAL_MODE}.json"

def invokeAgent(query, session_id, session_state=dict()):
    end_session: bool = False

    agentResponse = client.invoke_agent(
        inputText=query,
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        enableTrace=True,
        endSession=end_session,
        sessionState=session_state,
    )

    event_stream = agentResponse["completion"]
    try:
        traces = []
        for event in event_stream:
            if "chunk" in event:
                data = event["chunk"]["bytes"]
                agent_answer = data.decode("utf8")
                end_event_received = True
                return agent_answer, traces
            elif "trace" in event:
                traces.append(event["trace"])
            else:
                raise Exception("unexpected event.", event)
        return agent_answer, traces
    except Exception as e:
        raise Exception("unexpected event.", e)

with open(INPUT_FILE, 'r') as f:
    EVAL_DATA = json.load(f)

metrics_config = {
    "general": [FactualCorrectness(llm=evaluator_llm)],
    "aggregation": [FactualCorrectness(llm=evaluator_llm)],
    "vqa": [SemanticSimilarity(embeddings=embeddings)],
    "outofscope": [FactualCorrectness(llm=evaluator_llm)],
    "search": [SemanticSimilarity(embeddings=embeddings)]
}

mode_metrics = metrics_config.get(EVAL_MODE)

samples = []

for conversation in EVAL_DATA:
    session_id = str(uuid.uuid4())

    query = conversation["query"]
    reference = conversation["reference"]
    topics = conversation.get("reference_topics", []) 
    
    agent_answer, traces = invokeAgent(query, session_id)
    
    samples.append(SingleTurnSample(
        user_input=query,
        response=agent_answer,
        reference=reference
    ))

    print(f"[{EVAL_MODE}] Ran turn: {query[:40]}...")

os.makedirs("results", exist_ok=True)

dataset = EvaluationDataset(samples=samples)
results = evaluate(dataset=dataset, metrics=mode_metrics)
results.to_pandas().to_csv(f"results/{EVAL_MODE}_results.csv", index=False)

df = pd.read_csv(f"results/{EVAL_MODE}_results.csv")

metric_name = df.columns[-1]
average_score = df[metric_name].mean()

print(f"Metric: {metric_name}")
print(f"Average: {average_score:.4f}")
print(f"Done. Saved results for {EVAL_MODE}.")