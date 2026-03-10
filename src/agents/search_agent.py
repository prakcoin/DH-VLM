from strands import Agent, tool
from strands_tools import retrieve
from strands.multiagent import GraphBuilder
from strands.multiagent.graph import GraphState
from strands.multiagent.base import Status
from strands.models import BedrockModel
from src.tools.search_tools import serper_search
import logging

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

KB_PROMPT = """
Role:
Retrieve proper information to be used in search, and for final verification.

Guidelines:
Once the relevant information is obtained, add it to the rest of the query. 
The only required information to add is the item name.
If some information is classified as not available or to be updated.
"""

SEARCH_PROMPT = """
Role:
Find current and past listings for items using web search.

Guidelines:
Limit searches strictly to Dior Homme AW04.
Always include season and collection identifiers if the user query is vague.
Discard replicas, inspired items, or unrelated pieces.
Include source URLs with every fact. Do not hallucinate invalid URLs.
"""

AGGREGATOR_PROMPT = """
Role:
Aggregate all web search results and filter out irrelevant or redundant results.

Guidelines:
Filter based on ground truth provided by the knowledge base agent. If a search result doesn't match the item based in the knowledge base, filter it out. 
Make sure retrieved results are from Dior Homme AW04.
Discard duplicate listings, replicas, inspired items, or unrelated pieces.
Provide listing URLs. If a URL cannot be verified as functional, discard it. Never guess a URL.
"""

def all_dependencies_complete(required_nodes: list[str]):
    """Factory function to create AND condition for multiple dependencies."""
    def check_all_complete(state: GraphState) -> bool:
        return all(
            node_id in state.results and state.results[node_id].status == Status.COMPLETED
            for node_id in required_nodes
        )
    return check_all_complete

@tool
def search_assistant(query: str) -> str:
    """
    Handle queries requiring web search.

    Args:
    query (str): A question requiring external web search.

    Returns:
    Textual response synthesizing information from web sources, including cited URLs where applicable.
    """
    try:
        #builder = GraphBuilder()

        kb_agent = Agent(model=bedrock_model,
            system_prompt=KB_PROMPT, tools=[retrieve])
        google_agent = Agent(model=bedrock_model,
            system_prompt=SEARCH_PROMPT, tools=[serper_search])
        aggregator_agent = Agent(model=bedrock_model,
            system_prompt=AGGREGATOR_PROMPT)

        kb_results = kb_agent(f"Retrieve relevant information based on this query: {query}")
        search_results = google_agent(f"{kb_results}")
        response = aggregator_agent(f"Filter out any redundant or irrelevant results from these search results: {search_results}. Base the relevancy on these ground truths: {kb_results}")

        return str(response)
    except Exception as e:
        return f"Error in search assistant: {str(e)}"