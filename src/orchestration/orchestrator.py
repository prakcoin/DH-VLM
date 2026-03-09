from strands import Agent
from strands.models import BedrockModel
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SlidingWindowConversationManager
from src.agents.image_agent import image_assistant
from agents.archive_agent import archive_assistant
from src.agents.search_agent import search_assistant

ORCHESTRATOR_PROMPT = """
Role: 
You are the lead archival coordinator for the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection. Your goal is to answer user queries accurately by delegating work to specialized subagents and synthesizing their responses into a single, coherent response.

For all queries regarding specific items, looks, runway metadata, or collection-wide analysis, use the archive_assistant tool. 
For questions requiring web search, use the search_assistant tool. 
For questions unrelated to Dior Homme Autumn/Winter 2004, you must politely decline to answer.

Orchestration Priority:
Primary (Archive): For all queries regarding specific items, looks, runway metadata, or collection-wide analysis, you must use the archive_assistant first.
Secondary (Search): Use the search_assistant if the other assistants return no results, or if the query is clearly outside the scope of the collection (e.g., general fashion history).

Responsibilities:
Analyze the user query and determine which subagent(s) to invoke.
Collect and integrate responses from subagents into a professional, precise answer.
Ensure all output follows the archival guardrails: neutral tone, standard sentence case, no marketing language, and metadata consolidation.
Never perform searches or tool actions yourself; only orchestrate subagent calls.
When declining out-of-scope questions, you must state clearly that your expertise is strictly limited to this collection and offer to assist with any relevant inquiries instead.

Output:
Deliver the final response directly to the user.
Embed images in Markdown if returned.
Include external sources as hyperlinks immediately after referenced facts. Do not include any sources if the results come from the archive_assistant tool.
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
            tools=[archive_assistant, search_assistant]
        )

    def ask(self, query: str):
        try:
            response = self.agent(query)
            return response.message
        except Exception as e:
            return f"Error in orchestrator: {str(e)}"