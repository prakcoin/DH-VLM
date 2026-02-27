import boto3
import os
from dotenv import load_dotenv

load_dotenv()

class DHAgent:
    def __init__(self):
        self.client = boto3.client(
            service_name="bedrock-agent-runtime",
            region_name=os.getenv("AWS_REGION")
        )
        self.agent_id = os.getenv("AGENT_ID")
        self.alias_id = os.getenv("ALIAS_ID")

    def invoke(self, prompt, session_id):
        response = self.client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.alias_id,
            sessionId=session_id,
            inputText=prompt,
        )
        for event in response.get("completion"):
            if 'chunk' in event:
                yield event["chunk"]["bytes"].decode()