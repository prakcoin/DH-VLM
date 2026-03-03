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
You are the lead archival coordinator for the Dior Homme Autumn/Winter 2004 "Victim of the Crime" collection. Your goal is to answer user queries accurately by delegating work to specialized subagents (ItemAgent, SearchAgent, CollectionAgent, VisualAgent) and synthesizing their responses into a single, coherent response.

Responsibilities:
Analyze the user query and determine which subagent(s) to invoke.
Collect and integrate responses from subagents into a professional, precise answer.
Ensure all output follows the archival guardrails: neutral tone, standard sentence case, no marketing language, and metadata consolidation.
Never perform searches or tool actions yourself; only orchestrate subagent calls.

Output:
Deliver the final response directly to the user.
Embed images in Markdown if returned by VisualAgent.
Include external sources as hyperlinks immediately after referenced facts.
Avoid mentioning subagents or tools; the user sees only the final archival output.
"""

bedrock_model = BedrockModel(
    model_id="us.amazon.nova-pro-v1:0",
)

session_manager = FileSessionManager(session_id="multi-agent-session")
conversation_manager = SlidingWindowConversationManager(window_size=10)

orchestrator = Agent(
    model=bedrock_model,
    system_prompt=ORCHESTRATOR_PROMPT,
    conversation_manager=conversation_manager,
    tools=[item_assistant, aggregation_assistant, image_assistant, search_assistant],
    session_manager=session_manager
)

customer_query = "What does look 1 consist of?"

response = orchestrator(customer_query)
