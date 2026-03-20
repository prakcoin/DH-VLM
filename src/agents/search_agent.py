from strands import Agent, tool, AgentSkills
from strands.models import BedrockModel
from src.tools.search_tools import listing_search, general_search

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

plugin = AgentSkills(skills="src/agents/skills/search_skills")

SEARCH_PROMPT = """
Role:
Provide verified market and historical context via web searches.

Guidelines:
Limit searches strictly to Dior Homme AW04.
When declining out-of-scope questions, you must state clearly that your expertise is strictly limited to this collection and offer to assist with any relevant inquiries instead.
Always include season and collection identifiers if the user query is vague.
For historical data, cite the source URL for every fact. For marketplace results, provide only the direct listing URL once per item.
Avoid mentioning subagents or tools; the user sees only the final archival output.
Address the query directly and exclusively. Do not provide tangential context, historical background, or related media unless specifically requested.
"""

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
        archive_agent = Agent(
            model=bedrock_model,
            system_prompt=SEARCH_PROMPT,
            tools=[general_search, listing_search],
            plugins=[plugin]
        )

        response = archive_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in search assistant: {str(e)}"