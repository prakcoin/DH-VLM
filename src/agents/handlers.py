from typing import TYPE_CHECKING, Any, Literal, cast
from pydantic import BaseModel, Field
from strands import Agent
from strands.vended_plugins.steering import Guide, ModelSteeringAction, Proceed, SteeringHandler
from strands.vended_plugins.steering.context_providers.ledger_provider import LedgerProvider
from strands.models import BedrockModel
from strands.types.content import Message
from PIL import Image
import os

if TYPE_CHECKING:
    from strands import Agent as AgentType

class ToneDecision(BaseModel):
    """Structured output for output evaluation."""

    decision: Literal["proceed", "guide"] = Field(
        description="Steering decision: 'proceed' to accept, 'guide' to provide feedback"
    )
    reason: str = Field(description="Clear explanation of the decision and any guidance provided")


class AgentSteeringHandler(SteeringHandler):
    """Steering handler that validates model responses meet guidelines."""

    name = "agent_output_steering"

    def __init__(self, system_prompt) -> None:
        """Initialize the model output steering handler."""
        super().__init__(context_providers=[LedgerProvider()])

        self._system_prompt = system_prompt

        self._model = BedrockModel(
            model_id="us.amazon.nova-2-lite-v1:0",
        )

    async def steer_after_model(
        self,
        *,
        agent: "AgentType",
        message: Message,
        stop_reason: Literal[
            "content_filtered",
            "end_turn",
            "guardrail_intervened",
            "interrupt",
            "max_tokens",
            "stop_sequence",
            "tool_use",
        ],
        **kwargs: Any,
    ) -> ModelSteeringAction:
        """Validate that model responses meet guidelines."""
        if stop_reason != "end_turn":
            return Proceed(reason="Not a final response")

        content = message.get("content", [])
        text = " ".join(block.get("text", "") for block in content if block.get("text"))
        if not text:
            return Proceed(reason="No text content to evaluate")

        steering_agent = Agent(system_prompt=self._system_prompt, model=self._model, callback_handler=None)
        result = steering_agent(f"Evaluate this message:\n\n{text}", structured_output_model=ToneDecision)
        decision: ToneDecision = cast(ToneDecision, result.structured_output)

        match decision.decision:
            case "proceed":
                return Proceed(reason=decision.reason)
            case "guide":
                guidance = f"""Your previous response was NOT shown to the user.
{decision.reason}
Please provide a new response."""
                return Guide(reason=guidance)
            case _:
                return Proceed(reason="Unknown decision, defaulting to proceed")
    
    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        tool_name = tool_use.get("name", "")
        args = tool_use.get("input", {})

        ctx = self.steering_context.data.get()
        ledger = ctx.get("ledger", {}).get("tool_calls", [])

        # --- WORKFLOW 1 & 3: KB RETRIEVAL LOGIC ---
        if tool_name == "retrieve":
            query = args.get("text", "").lower()
            forbidden = ["dior", "homme", "aw04", "autumn", "winter", "2004"]
            if any(word in query for word in forbidden):
                return Guide(reason="Remove brand names/seasons from tool input. Use core subject only.")

        # --- WORKFLOW 1: LOOK COMPOSITION SEQUENCE ---
        if tool_name == "get_look_composition":
            look_num = args.get("look_number")

            if not look_num:
                return Guide(reason="The look number is missing. Please provide it, either through knowledge base retrieval or the user's query.")

            look_str = str(look_num).strip()

            if not look_str.isdigit():
                return Guide(
                    reason=f"The look number '{look_str}' is invalid. It must be a "
                        "positive integer (e.g., 5). No decimals or words."
                )

            if int(look_str) <= 0 or int(look_str) > 45:
                return Guide(reason="The look number must be between 1-45 inclusive, as there are only 45 looks in the collection.")

        if tool_name == "get_image_details":
            filenames = args.get("image_filenames")

            if not isinstance(filenames, list):
                return Guide(
                    reason="The 'image_filenames' argument must be a LIST of strings, "
                        "even if there is only one image."
                )

            if len(filenames) == 0:
                return Guide(reason="You must provide at least one filename to analyze.")

            valid_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
            for f in filenames:
                if not str(f).lower().endswith(valid_exts):
                    return Guide(reason=f"Invalid extension. Use: {', '.join(valid_exts)}")

        # --- WORKFLOW 2: IMAGE VALIDATION ---
        if tool_name == "image_retrieve":
            val = args.get("image_path")
            valid_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")

            if not val or not os.path.exists(str(val)):
                return Guide(reason="Image path does not exist.")

            if not str(val).lower().endswith(valid_exts):
                return Guide(reason=f"Invalid extension. Use: {', '.join(valid_exts)}")

            try:
                with Image.open(val) as img:
                    img.verify()
            except Exception:
                return Guide(reason="The file is corrupted or not a valid image.")
            
        if tool_name == "get_cloudfront_url":
            ir_success = any(c["tool_name"] == "image_retrieve" and c["status"] == "success" for c in ledger)
            if not ir_success:
                return Guide(reason="Prerequisite tool missing. You must successfully call 'image_retrieve' to get images before using this tool.")

        if tool_name == "get_image_comparison":
            if args.get("query_filename") == args.get("retrieved_filename"):
                return Guide(reason="Comparison requires two different images. Duplicate images were provided.")

            valid_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")

            query_filename = args.get("query_filename")
            if not query_filename or not os.path.exists(str(query_filename)):
                return Guide(reason="Query image path does not exist.")
            if not str(query_filename).lower().endswith(valid_exts):
                return Guide(reason=f"Invalid extension. Use: {', '.join(valid_exts)}")
            try:
                with Image.open(query_filename) as img:
                    img.verify()
            except Exception:
                return Guide(reason="The file is corrupted or not a valid image.")

            retrieved_filename = args.get("retrieved_filename")
            if not retrieved_filename or not str(retrieved_filename).lower().endswith(valid_exts):
                return Guide(reason=f"Invalid or missing retrieved filename. Use: {', '.join(valid_exts)}")

        return Proceed(reason="Tool input matches workflow requirements.")