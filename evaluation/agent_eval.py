import os
import uuid
import json
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agent import DHAgent
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
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

agent = DHAgent()
bedrock_llm = ChatBedrock(model_id="us.amazon.nova-pro-v1:0", region_name="us-east-1")
evaluator_llm = LangchainLLMWrapper(bedrock_llm)
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.nova-2-multimodal-embeddings-v1:0", region_name="us-east-1")
embeddings = LangchainEmbeddingsWrapper(bedrock_embeddings)

EVAL_MODE = "general"
INPUT_FILE = f"datasets/eval_{EVAL_MODE}.json"
baseline_system_prompt = """Role: 
    You are the lead archival assistant for the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection. Your goal is to provide precise and factual information regarding this specific collection.

    Archival Guardrails:
    Every sentence must be in standard sentence case (e.g., 'The jacket is leather,' not 'THE JACKET IS LEATHER' or 'The Jacket Is Leather').
    Use a neutral, professional, editorial tone. Avoid marketing language.
    All look-based data MUST be sorted by Look Number in ascending order (1, 2, 3...).
    When summarizing motifs, you must distinguish between "Baseline Attributes" and "Collection-Specific Design Language." Do not report "Solid pattern," "Cotton," or "Buttons" as recurring motifs unless specifically asked. These are considered functional/ubiquitous and do not constitute a motif for this collection. Only report a recurring element if it serves as a stylistic signature. A "belt" is a utility; a "belt with onyx studs" is a motif. If an attribute appears in nearly every look (like "Solid"), it is a baseline characteristic and should be omitted from a summary of motifs.
    When multiple results refer to the same garment, consolidate them into a single entry. List the item name once, followed by all look numbers where it appears in parentheses. (e.g., "Leather studded belt (Looks 5, 7, 9, 12, 16...)").

    Output Format:
    Your response must contain only the direct answer to the user's query. If the user asks for "Look Numbers," do not provide "Materials." If the user asks for "Items," do not provide "Reference Codes." Every extra word of metadata is a failure of the archival protocol.
    Never list the exact same item name more than once in a single response. Use a comma-separated list for look numbers to keep the response concise and professional."""

def invokeAgent(query, session_id):
    full_text = ""
    try:
        for chunk in agent.invoke(query, session_id):
            full_text += chunk
        return full_text if full_text else "Agent failed to provide a response."
    except Exception as e:
        return f"Error: {str(e)}"

def invokeBaseline(query):
    messages = [
        SystemMessage(content=baseline_system_prompt),
        HumanMessage(content=query)
    ]
    
    response = bedrock_llm.invoke(messages)
    
    return response.content

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

agent_samples = []
agent_answers = []
baseline_samples = []
baseline_answers = []

for conversation in EVAL_DATA:
    session_id = str(uuid.uuid4())

    query = conversation["query"]
    reference = conversation["reference"]
    topics = conversation.get("reference_topics", []) 
    
    agent_answer = invokeAgent(query, session_id)
    baseline_answer = invokeBaseline(query)
    
    agent_samples.append(SingleTurnSample(
        user_input=query,
        response=agent_answer,
        reference=reference
    ))

    agent_answers.append(agent_answer)
    baseline_answers.append(baseline_answer)

    print(f"[{EVAL_MODE}] Ran turn: {query[:40]}...")

os.makedirs("results", exist_ok=True)

dataset = EvaluationDataset(samples=agent_samples)
results = evaluate(dataset=dataset, metrics=mode_metrics)
results.to_pandas().to_csv(f"results/{EVAL_MODE}_results.csv", index=False)

df = pd.read_csv(f"results/{EVAL_MODE}_results.csv")

metric_name = df.columns[-1]
average_score = df[metric_name].mean()

print(f"Metric: {metric_name}")
print(f"Average: {average_score:.4f}")
print(f"Done. Saved results for {EVAL_MODE}.")

for agent_answer, baseline_answer in zip(agent_answers, baseline_answers):
    print("Agent Answer:\n", agent_answer)
    print("Baseline Answer:\n", baseline_answer)