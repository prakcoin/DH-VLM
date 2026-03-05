import boto3

client = boto3.client('bedrock-agentcore-control')

response = client.update_agent_runtime(
    agentRuntimeId='aw04_agent_runtime-amGOL14SZM',
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': '397172001076.dkr.ecr.us-east-1.amazonaws.com/aw04-agent:latest'
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn='arn:aws:iam::397172001076:role/AgentRuntimeRole'
)

print(f"Agent Runtime update initiated!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")