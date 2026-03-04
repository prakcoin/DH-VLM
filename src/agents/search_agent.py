from strands import Agent, tool
from strands.models import BedrockModel
from src.tools.search_tools import serper_search, tavily_search

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)


SEARCH_PROMPT = """
Role:
Provide verified market and historical context via web searches.

Guidelines:
Limit searches strictly to Dior Homme AW04.
Always include season and collection identifiers if the user query is vague.
Discard replicas, inspired items, or unrelated pieces.
Include source URLs with every fact.
"""

@tool
def search_assistant(query: str) -> str:
    """
    Handle queries requiring web search.

    Use this as a conversational agent for performing web searches for details unseen in the metadata.

    Args:
    query (str): A question requiring external web search.

    Returns:
    Textual response synthesizing information from web sources, including cited URLs where applicable.
    """
    try:
        search_agent = Agent(
            model=bedrock_model,
            system_prompt=SEARCH_PROMPT,
            tools=[serper_search, tavily_search]
        )

        response = search_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in search assistant: {str(e)}"