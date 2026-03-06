import boto3

client = boto3.client('bedrock-agentcore-control')

response = client.create_agent_runtime(
    agentRuntimeName='aw04_agent_runtime',
    agentRuntimeArtifact={
        'containerConfiguration': {
            'containerUri': '397172001076.dkr.ecr.us-east-1.amazonaws.com/aw04-agent:latest'
        }
    },
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn='arn:aws:iam::397172001076:role/AgentRuntimeRole'
)

print(f"Agent Runtime created successfully!")
print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
print(f"Status: {response['status']}")