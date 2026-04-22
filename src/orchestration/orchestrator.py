from strands import Agent
from strands.models import BedrockModel
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SlidingWindowConversationManager, SummarizingConversationManager
from src.agents.archive_agent import archive_assistant
from src.agents.search_agent import search_assistant
from src.agents.conversation_managers import ProactiveSummarizingConversationManager
from src.agents.hooks import NotifyOnlyGuardrailsHook, LimitToolCounts
from src.agents.handlers import AgentSteeringHandler

ORCHESTRATOR_PROMPT = """
Role: 
You are the lead archival coordinator for the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection. Your goal is to answer user queries accurately by delegating work to specialized subagents and synthesizing their responses into a single, coherent response.

Guidelines:
The knowledge base contains runway look breakdowns: garment names, reference codes, materials, colors, patterns, construction notes, and images. The Additional Notes field also documents non-runway variants for select items — alternate colorways, pieces not featured on the runway, alternate reference codes, and sizing differences. It does not contain pricing, cultural context, editorial analysis, celebrity associations, or hardware brand details.
Route each query to the single most appropriate subagent. Do not call archive_assistant as a first step for queries that clearly require search.
Use archive_assistant for specific runway items, look compositions, garment descriptions, attributes such as materials, colors, reference codes, collection-wide inventory, and non-runway variants or alternate versions of items.
Use search_assistant directly (without calling archive_assistant first) for marketplace listings, resale prices, current availability, pricing guidance, who wore a piece, hardware or component brands, and collection context such as music, theming, cultural impact, design inspirations, editorial commentary, or press coverage.
"""

handler = AgentSteeringHandler(
    system_prompt="""
    You are providing guidance to ensure proper formatting of information.

    Guidance:
    Maintain a neutral tone.
    Do not include internal monologues, reasoning steps, or tags like <thinking>. Avoid mentioning subagents or tools.
    
    When subagents return their analysis, evaluate the combined text and deliver the final response directly to the user.
    """
)

class Orchestrator:
    """Wrapper class for the multi-agent orchestration system."""

    def __init__(self):
        self.model = BedrockModel(model_id="us.amazon.nova-2-lite-v1:0",
                                  temperature=0.0,
                                  max_tokens=12000)
                                #   guardrail_id="ys4jzzz12h6r",
                                #   guardrail_version="14",
                                #   guardrail_trace="enabled")
        #self.session_manager = FileSessionManager(session_id='new-session')
        self.conversation_manager = ProactiveSummarizingConversationManager() #SlidingWindowConversationManager(window_size=20, should_truncate_results=True, per_turn=5)
        self.agent = Agent(
            model=self.model,
            system_prompt=ORCHESTRATOR_PROMPT,
            #session_manager=self.session_manager,
            conversation_manager=self.conversation_manager,
            callback_handler=None,
            tools=[archive_assistant, search_assistant],
            hooks=[NotifyOnlyGuardrailsHook("ys4jzzz12h6r", "14"), LimitToolCounts(max_tool_counts={"archive_assistant": 3, "search_assistant": 3})],
            plugins=[handler]
        )

    def ask(self, query: str):
        try:
            response = self.agent(query)
            print(f"\n{'='*60}")
            print(f"Response Summary")
            print(f"{'='*60}")
            print(response.metrics.get_summary())
            if response.stop_reason == "guardrail_intervened":
                return "Sorry, this question is out of scope. This archive is strictly dedicated to Dior Homme Autumn/Winter 2004."
            return response.message
        except Exception as e:
            return f"Error in orchestrator: {str(e)}"