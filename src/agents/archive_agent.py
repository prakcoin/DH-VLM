from strands import Agent, tool, AgentSkills
from strands.models import BedrockModel
from src.tools.archive_tools import get_look_composition, get_collection_summary, get_visual_confirmation, get_image_input
from strands_tools import retrieve
from src.agents.hooks import LimitToolCounts

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-pro-v1:0",
)

plugin = AgentSkills(skills="src/agents/skills/archive_skills")

PROMPT = """
Role:
Your role is to provide precise data on individual items, specific looks, and collection-wide analysis by utilizing the archival toolset.

Guidelines:
When providing an answer, always prioritize the knowledge base metadata; if specific details cannot be retrieved, use visual analysis through to confirm information.
Consolidate duplicate entries.
Do not hallucinate item information. All information must be derived from the archive.
Do not include any sources in your response.
Avoid mentioning subagents or tools; the user sees only the final archival output.
Address the query directly and exclusively. Do not provide tangential context, historical background, or related media unless specifically requested.
"""

@tool
def archive_assistant(query: str) -> str:
    """
    Handle all knowledge base queries related to the Dior Homme AW04 "Victim of the Crime" collection.
    
    Args:
    query (str): A question about an item.

    Returns: 
    Textual response synthesized from internal archival tools.
    """
    try:
        limit_hook = LimitToolCounts(max_tool_counts={"retrieve": 3})

        archive_agent = Agent(
            model=bedrock_model,
            system_prompt=PROMPT,
            tools=[get_look_composition, get_collection_summary, get_visual_confirmation, get_image_input, retrieve],
            plugins=[plugin],
            hooks=[limit_hook]
        )

        response = archive_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in item assistant: {str(e)}"