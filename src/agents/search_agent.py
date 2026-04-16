from strands import Agent, tool, AgentSkills
from strands.models import BedrockModel
from src.tools.search_tools.general_search import general_search 
from src.tools.search_tools.listing_search import listing_search
from src.agents.handlers import AgentSteeringHandler
from src.agents.hooks import LimitToolCounts
import os

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-2-lite-v1:0",
    temperature=0.0,
    max_tokens=12000
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) 
path = os.path.join(ROOT_DIR, "src/agents/skills/search_skills")

plugin = AgentSkills(skills=path)

SEARCH_PROMPT = """
Role:
Provide verified market and historical context via web searches.
"""

handler = AgentSteeringHandler(
    system_prompt="""
    You are providing guidance to ensure proper formatting of information.

    Guidance:
    For historical data, cite the source URL for every fact. For marketplace results, provide only the direct listing URL once per item.
    
    When the tools return their responses, evaluate the text and deliver the final response directly to the user.
    """
)

@tool
def search_assistant(query: str) -> str:
    """
    Handle queries requiring web search.

    Args:
    query (str): A question requiring external web search.

    Returns:
    Textual response synthesizing information from web sources, including cited URLs where applicable.
    """
    limit_hook = LimitToolCounts(max_tool_counts={"general_search": 3, "listing_search": 3})
    try:
        archive_agent = Agent(
            model=bedrock_model,
            system_prompt=SEARCH_PROMPT,
            tools=[general_search, listing_search],
            plugins=[plugin, handler],
            hooks=[limit_hook],
            callback_handler=None
        )

        response = archive_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in search assistant: {str(e)}"