from strands import Agent, tool
from strands.models import BedrockModel
from src.tools.item_tools import get_look_images, get_look_composition, get_item_attribute

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

ITEM_PROMPT = """
Role:
Handle questions about individual items, looks, and metadata.

Guidelines:
Never ask the user for look numbers; retrieve directly from the database.
Pass image filenames to VisualAgent if detailed visual analysis is requested.
"""

@tool
def item_assistant(query: str) -> str:
    """
    Handle queries about single items, looks, and their metadata.

    Use this as a conversational agent to answer questions about specific items, retrieve look compositions, or provide images.

    Args:
    query (str): A question about an item.

    Returns: 
    Textual response with item information.
    """
    try:
        item_agent = Agent(
            model=bedrock_model,
            system_prompt=ITEM_PROMPT,
            tools=[get_look_images, get_look_composition, get_item_attribute]
        )

        response = item_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in item assistant: {str(e)}"