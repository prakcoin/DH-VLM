from strands import Agent, tool
from strands.models import BedrockModel
from src.tools.image_tools import get_image_details

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

IMAGE_PROMPT = """
Role:
Analyze look images for fit, silhouette, texture, and aesthetic details.

Guidelines:
Always retrieve filenames via ItemAgent.
Combine visual analysis with metadata for the final answer.
Report discrepancies between visual and metadata observations.
"""

@tool
def image_assistant(query: str) -> str:
    """
    Handle queries requiring visual analysis of look images.

    Use this as a conversational agent to answer questions about the visual appearance of garments, including fit, texture, patterns, and specific design details that are better understood visually than through text.
    
    Args:
    query (str): A question requiring visual inspection of a look or garment.

    Returns: 
    Textual response regarding visual analysis of images.
    """
    try:
        image_agent = Agent(
            model=bedrock_model,
            system_prompt=IMAGE_PROMPT,
            tools=[get_image_details]
        )

        response = image_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in image assistant: {str(e)}"