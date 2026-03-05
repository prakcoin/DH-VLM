from strands import Agent
from strands.models import BedrockModel
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SlidingWindowConversationManager
from src.agents.aggregation_agent import aggregation_assistant 
from src.agents.image_agent import image_assistant
from src.agents.item_agent import item_assistant
from src.agents.search_agent import search_assistant

ORCHESTRATOR_PROMPT = """
Role: 
You are the lead archival coordinator for the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection. Your goal is to answer user queries accurately by delegating work to specialized subagents and synthesizing their responses into a single, coherent response.

For questions involving questions about individual items, looks, and metadata, use the item_assistant tool.
For questions involving aggregation or analysis across the entire collection, use the aggregation_assistant tool.
For questions requiring web search, use the search_assistant tool.
For questions requiring visual analysis, use the image_assistant tool. Always retrieve the image filenames first with the item_assistant tool using get_look_images, then pass them into image_assistant. Never ask the user for filenames.

Responsibilities:
Analyze the user query and determine which subagent(s) to invoke.
Collect and integrate responses from subagents into a professional, precise answer.
Ensure all output follows the archival guardrails: neutral tone, standard sentence case, no marketing language, and metadata consolidation.
Never perform searches or tool actions yourself; only orchestrate subagent calls.

Output:
Deliver the final response directly to the user.
Embed images in Markdown if returned.
Include external sources as hyperlinks immediately after referenced facts.
Avoid mentioning subagents or tools; the user sees only the final archival output.
"""

class Orchestrator:
    """Wrapper class for the multi-agent orchestration system."""

    def __init__(self):
        self.model = BedrockModel(model_id="us.amazon.nova-pro-v1:0")

        self.conversation_manager = SlidingWindowConversationManager(window_size=10)

        self.agent = Agent(
            model=self.model,
            system_prompt=ORCHESTRATOR_PROMPT,
            conversation_manager=self.conversation_manager,
            callback_handler=None,
            tools=[item_assistant, aggregation_assistant, image_assistant, search_assistant]
        )

    def ask(self, query: str):
        try:
            response = self.agent(query)
            return response.message
        except Exception as e:
            return f"Error in orchestrator: {str(e)}"