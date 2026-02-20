import boto3
import os
import uuid
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics import ContextPrecision, Faithfulness
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample, EvaluationDataset
from ragas.integrations.amazon_bedrock import convert_to_ragas_messages
from ragas.messages import HumanMessage
from ragas import evaluate
from dotenv import load_dotenv

load_dotenv()
client = boto3.client("bedrock-agent-runtime")
agent_id = os.getenv("AGENT_ID")
alias_id = os.getenv("ALIAS_ID")

evaluator_llm = llm_factory('gpt-4o-mini', client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

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

EVAL_DATA = [
    {
        "query": "What does look 1 consist of?",
        "reference": "Look 1 consists of a 1B blazer, a leather tie, a pinstripe shirt, trousers, suede moto boots, aviator sunglasses, a leather belt, and a bandana bracelet."
    },
    {
        "query": "What is the reference code for the beetle leather jacket?",
        "reference": "The reference code for the beetle leather jacket is 4HH5041101."
    },
    {
        "query": "What material is the jacket in look 2 made from?",
        "reference": "The jacket in look 2 is made from leather, more specifically calfskin."
    },
    {
        "query": "What type of jacket is featured in look 28?",
        "reference": "Look 28 features a belted blazer."
    },
    {
        "query": "Does look 17 include any accessories?",
        "reference": "Yes. Look 17 includes a single-split scarf, a leather belt, a suede messenger bag, and a bandana bracelet."
    },
    {
        "query": "Besides the standard reference code, what are the alternative codes for the whiskered trousers?",
        "reference": "Alternative reference codes for the whiskered trousers include 4SH1016284 and 4SH1016684, where the 4SH prefix denotes pre-fall."
    },
    {
        "query": "Which look features the beetle leather vest?",
        "reference": "The beetle leather vest is featured in Look 43."
    },
    {
        "query": "What is unique about the design and inseam of the whiskered trousers?",
        "reference": "The whiskered trousers feature a whiskering effect near the crotch and pockets, and unlike other trousers in the collection, they have an extended inseam similar to standard Dior Homme jeans."
    }
]

samples = []

for item in EVAL_DATA:
    query = item["query"]
    reference = item["reference"]
    
    agent_answer, traces = invokeAgent(query, str(uuid.uuid4()))
    
    retrieved_contexts = []
    for trace in traces:
        if "orchestrationTrace" in trace:
            orch = trace["orchestrationTrace"]
            if "knowledgeBaseLookupOutput" in orch:
                for ref in orch["knowledgeBaseLookupOutput"].get("retrievedReferences", []):
                    retrieved_contexts.append(ref["content"]["text"])
    
    samples.append(SingleTurnSample(
        user_input=query,
        response=agent_answer,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    ))

dataset = EvaluationDataset(samples=samples)

# Run evaluation
result = evaluate(
    dataset=dataset,
    metrics=[ContextPrecision(llm=evaluator_llm), Faithfulness(llm=evaluator_llm)],
)

print(result.to_pandas())