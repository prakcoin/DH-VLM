from strands import Agent, tool, AgentSkills
from strands.models import BedrockModel
from src.tools.archive_tools.collection_inventory import get_collection_inventory
from src.tools.archive_tools.image_input import get_image_input
from src.tools.archive_tools.look_analysis import get_look_analysis
from strands_tools import retrieve
from src.agents.hooks import LimitToolCounts
from src.agents.handlers import AgentSteeringHandler
import os

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-2-lite-v1:0",
    temperature=0.0,
    max_tokens=12000
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) 
path = os.path.join(ROOT_DIR, "src/agents/skills/archive_skills")

plugin = AgentSkills(skills=path)

PROMPT = """
Role:
Provide precise data on individual items, specific looks, and collection-wide analysis by utilizing the archival toolset.

Guidelines:
All information must be derived directly from the archive tools. Do not infer, guess, or fabricate.
The knowledge base is runway data: look-by-look breakdowns of every piece present on the runway. Each garment entry contains only: item name, reference code, look number, category, subcategory, colors, pattern, materials, additional construction notes, and image filenames.
If a question requires information beyond these fields — such as hardware brands, who wore a piece off-runway, pricing, cultural context, or editorial analysis — respond with "This information is not documented in the archive" rather than attempting to answer.
"""

handler = AgentSteeringHandler(
    system_prompt="""
    You are providing guidance to ensure proper formatting of information.

    Guidance:
    Consolidate duplicate entries.
    Do not provide tangential context, historical background, or related media unless specifically requested.

    When the tools return their responses, evaluate the text and deliver the final response directly to the user.
    """
)

@tool
def archive_assistant(query: str) -> str:
    """
    Handle all knowledge base queries related to the Dior Homme AW04 "Victim of the Crime" collection.
    
    Args:
    query (str): A question about an item.

    Returns: 
    Textual response synthesized from internal archival tools.
    """
    limit_hook = LimitToolCounts(max_tool_counts={"retrieve": 3, "get_look_analysis": 3, "get_collection_inventory": 3, "get_image_input": 3})

    try:
        archive_agent = Agent(
            model=bedrock_model,
            system_prompt=PROMPT,
            tools=[get_collection_inventory, get_look_analysis, get_image_input, retrieve],
            plugins=[plugin, handler],
            hooks=[limit_hook],
            callback_handler=None
        )

        response = archive_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in item assistant: {str(e)}"