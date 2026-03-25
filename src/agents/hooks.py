import boto3
from strands import tool
from strands.hooks import HookRegistry, HookProvider, BeforeToolCallEvent, BeforeInvocationEvent, MessageAddedEvent, AfterInvocationEvent, AfterNodeCallEvent
from threading import Lock

class LimitToolCounts(HookProvider):
    """Limits the number of times tools can be called per agent invocation"""

    def __init__(self, max_tool_counts: dict[str, int]):
        """
        Initializer.

        Args:
            max_tool_counts: A dictionary mapping tool names to max call counts for
                tools. If a tool is not specified in it, the tool can be called as many
                times as desired
        """
        self.max_tool_counts = max_tool_counts
        self.tool_counts = {}
        self._lock = Lock()

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeInvocationEvent, self.reset_counts)
        registry.add_callback(BeforeToolCallEvent, self.intercept_tool)

    def reset_counts(self, event: BeforeInvocationEvent) -> None:
        with self._lock:
            self.tool_counts = {}

    def intercept_tool(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use["name"]
        with self._lock:
            max_tool_count = self.max_tool_counts.get(tool_name)
            tool_count = self.tool_counts.get(tool_name, 0) + 1
            self.tool_counts[tool_name] = tool_count

        if max_tool_count and tool_count > max_tool_count:
            event.cancel_tool = (
                f"Tool '{tool_name}' has been invoked too many and is now being throttled. "
                f"DO NOT CALL THIS TOOL ANYMORE "
            )

class ForceSingleExecutionHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterNodeCallEvent, self.terminate_after_call)

    def terminate_after_call(self, event: AfterNodeCallEvent) -> None:
        request_state = event.invocation_state.get("request_state", {})
        
        request_state["stop_event_loop"] = True
        
        tool_name = event.tool_use.get("name", "unknown tool")
        print(f"Path Locked: '{tool_name}' completed. stop_event_loop set to True.")

class NotifyOnlyGuardrailsHook(HookProvider):
    def __init__(self, guardrail_id: str, guardrail_version: str):
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.bedrock_client = boto3.client("bedrock-runtime", "us-east-1") # change to your AWS region

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(MessageAddedEvent, self.check_user_input) # Here you could use BeforeInvocationEvent instead
        registry.add_callback(AfterInvocationEvent, self.check_assistant_response)

    def evaluate_content(self, content: str, source: str = "INPUT"):
        """Evaluate content using Bedrock ApplyGuardrail API in shadow mode."""
        try:
            response = self.bedrock_client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source=source,
                content=[{"text": {"text": content}}]
            )

            if response.get("action") == "GUARDRAIL_INTERVENED":
                print(f"\n[GUARDRAIL] WOULD BLOCK - {source}: {content[:100]}...")
                # Show violation details from assessments
                for assessment in response.get("assessments", []):
                    print(f"Assessment: {assessment}")
                    if "topicPolicy" in assessment:
                        for topic in assessment["topicPolicy"].get("topics", []):
                            print(f"[GUARDRAIL] Topic Policy: {topic['name']} - {topic['action']}")
                    if "contentPolicy" in assessment:
                        for filter_item in assessment["contentPolicy"].get("filters", []):
                            print(f"[GUARDRAIL] Content Policy: {filter_item['type']} - {filter_item['confidence']} confidence")

        except Exception as e:
            print(f"[GUARDRAIL] Evaluation failed: {e}")

    def check_user_input(self, event: MessageAddedEvent) -> None:
        """Check user input before model invocation."""
        if event.message.get("role") == "user":
            content = "".join(block.get("text", "") for block in event.message.get("content", []))
            if content:
                self.evaluate_content(content, "INPUT")

    def check_assistant_response(self, event: AfterInvocationEvent) -> None:
        """Check assistant response after model invocation with delay to avoid interrupting output."""
        if event.agent.messages and event.agent.messages[-1].get("role") == "assistant":
            assistant_message = event.agent.messages[-1]
            content = "".join(block.get("text", "") for block in assistant_message.get("content", []))
            if content:
                self.evaluate_content(content, "OUTPUT")