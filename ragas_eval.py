import boto3
import os
import uuid
import time
import json
from langchain_aws import ChatBedrock
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, ToolCallAccuracy, ToolCallF1
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample, EvaluationDataset
from ragas.integrations.amazon_bedrock import convert_to_ragas_messages
from ragas.messages import ToolCall
from ragas import evaluate
from dotenv import load_dotenv

load_dotenv()
client = boto3.client("bedrock-agent-runtime")
agent_id = os.getenv("AGENT_ID")
alias_id = os.getenv("ALIAS_ID")

model_id = "us.amazon.nova-pro-v1:0"
region_name = "us-east-1"

bedrock_llm = ChatBedrock(model_id=model_id, region_name=region_name)
evaluator_llm = LangchainLLMWrapper(bedrock_llm)

def invokeAgent(query, session_id, session_state=dict()):
    end_session: bool = False

    # invoke the agent API
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

with open('eval.json', 'r') as f:
    EVAL_DATA = json.load(f)

multi_samples = []
single_samples = []

for item in EVAL_DATA:
    query = item["query"]
    reference = item["reference"]
    
    agent_answer, traces = invokeAgent(query, str(uuid.uuid4()))
    ragas_messages = convert_to_ragas_messages(traces)
    ref_tool_calls = [
        ToolCall(name=tool["name"], args=tool["args"]) 
        for tool in item.get("expected_tools", [])
    ]

    single_samples.append(SingleTurnSample(
        response=agent_answer,
        reference=reference,
    ))

    multi_samples.append(MultiTurnSample(
        user_input=ragas_messages,
        response=agent_answer,
        reference=reference,
        reference_tool_calls=ref_tool_calls
    ))
    print("Query: ", query)
    print("Ground Truth: ", reference)
    print("Agent Answer: ", agent_answer)
    time.sleep(2.5)    

multi_dataset = EvaluationDataset(samples=multi_samples)
single_dataset = EvaluationDataset(samples=single_samples)

single_result = evaluate(
    dataset=single_dataset,
    metrics=[FactualCorrectness(llm=evaluator_llm)],
)

multi_result = evaluate(
    dataset=multi_dataset,
    metrics=[ToolCallAccuracy(), ToolCallF1()],
)

single_df = single_result.to_pandas()
single_df.to_csv("single_results.csv", index=False)
multi_df = multi_result.to_pandas()
multi_df.to_csv("multi_results.csv", index=False)
print("Results saved")