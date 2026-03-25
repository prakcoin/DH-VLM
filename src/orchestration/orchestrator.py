from strands import Agent
from strands.models import BedrockModel
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SlidingWindowConversationManager
from agents.archive_agent import archive_assistant
from src.agents.search_agent import search_assistant
from src.agents.hooks import NotifyOnlyGuardrailsHook

ORCHESTRATOR_PROMPT = """
Role: 
You are the lead archival coordinator for the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection. Your goal is to answer user queries accurately by delegating work to specialized subagents and synthesizing their responses into a single, coherent response.

Analyze the user query and determine which subagent(s) to invoke:
1. For all queries regarding specific items, looks, visual analysis, or collection-wide analysis, use the archive_assistant tool. 
2. For questions requiring web search (such as for listings or pricing, or for information not documented in the knowledge base such as music or theming), use the search_assistant tool. 

Responsibilities:
Collect and integrate responses from subagents into a professional, precise answer.
Ensure all output follows the archival guardrails: neutral tone, standard sentence case, no marketing language, and metadata consolidation.
Never perform searches or tool actions yourself; only orchestrate subagent calls.

Output:
Deliver the final response directly to the user.
DO NOT include internal monologues, reasoning steps, or tags like <thinking>.
"""

class Orchestrator:
    """Wrapper class for the multi-agent orchestration system."""

    def __init__(self):
        self.model = BedrockModel(model_id="us.amazon.nova-pro-v1:0",
                                  guardrail_id="ys4jzzz12h6r",
                                  guardrail_version="14",
                                  guardrail_trace="enabled")
        #self.session_manager = FileSessionManager(session_id='new-session')
        self.conversation_manager = SlidingWindowConversationManager(window_size=10)

        self.agent = Agent(
            model=self.model,
            system_prompt=ORCHESTRATOR_PROMPT,
            #session_manager=self.session_manager,
            conversation_manager=self.conversation_manager,
            callback_handler=None,
            tools=[archive_assistant, search_assistant],
            hooks=[NotifyOnlyGuardrailsHook("ys4jzzz12h6r", "14")]
        )

    def ask(self, query: str):
        try:
            response = self.agent(query)
            if response.stop_reason == "guardrail_intervened":
                return "Sorry, this question is out of scope. This archive is strictly dedicated to Dior Homme Autumn/Winter 2004."
            return response.message
        except Exception as e:
            return f"Error in orchestrator: {str(e)}"