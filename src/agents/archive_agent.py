from strands import Agent, tool
from strands.models import BedrockModel
from tools.archive_tools import get_look_composition, get_collection_summary, get_item_counts
from tools.image_tools import get_look_images, get_image_details, image_workflow
from strands_tools import retrieve

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-lite-v1:0",
)

PROMPT = """
Role:
Your role is to provide precise data on individual items, specific looks, and collection-wide analysis by utilizing the archival toolset.

Use the retrieve tool for both individual item details (reference codes, materials, design features) and broader multi-item searches. When using this, pass only the relevant terms (item name or metadata) rather than the full user query. Never pass look numbers.
To get the full clothing item composition of a look, use the get_look_composition tool. Only use it when you have a look number, and when asked what a look consists of. 
To retrieve the runway images of a certain look, use the get_look_images tool. Only use it when you have a look number, and when asked to retrieve runway look images.
To get a full collection summary, use the get_collection_summary tool. Do not pass any parameters.
To answer queries involving visual analysis or identifying visual characteristics (e.g. a query stating "based on the images"), use the image_workflow tool. Pass the relevant aspects of the query that need to be analyzed (e.g. an item, look, or feature) to the tool, rather than the full query.

Guidelines:
Exclude generic functional components (buttons, belts, solids) unless explicitly asked.
Consolidate duplicate entries.
Never ask the user for look numbers.
When tools require look numbers, do not use them when you don't have one.
If retrieved results yield a low score, retry with a lower threshold. If results remain below the threshold, return them anyway but state that they are provided with lower confidence.
All data within the archival toolset is exclusively from the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection. There is no data from any other collection.
When performing visual analysis, report discrepancies between visual and metadata observations.
Always prioritize the knowledge base metadata; if specific details cannot be retrieved, use visual analysis through to confirm information.
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
        archive_agent = Agent(
            model=bedrock_model,
            system_prompt=PROMPT,
            tools=[get_look_composition, get_look_images, get_collection_summary, image_workflow, retrieve]
        )

        response = archive_agent(query)
        return str(response)
    except Exception as e:
        return f"Error in item assistant: {str(e)}"