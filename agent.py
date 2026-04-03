import os
import boto3
import json
from fastapi import FastAPI, HTTPException, Request
from datetime import datetime,timezone
import logging

logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

from strands.telemetry import StrandsTelemetry

strands_telemetry = StrandsTelemetry()
strands_telemetry.setup_console_exporter()
strands_telemetry.setup_otlp_exporter()
strands_telemetry.setup_meter(
    enable_console_exporter=True,
    enable_otlp_exporter=True)

def load_secrets():
    secret_name = "dh-agent/config"
    region_name = "us-east-1"

    client = boto3.client('secretsmanager', region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)
    secrets = json.loads(response['SecretString'])

    for key, value in secrets.items():
        os.environ[key] = str(value)

load_secrets()

from src.orchestration.orchestrator import Orchestrator

app = FastAPI(title="DH-Agent Server", version="1.0.0")
agent = Orchestrator()

@app.post("/invocations")
async def invoke_agent(request: Request):
    try:
        body = await request.json()
        user_message = body.get("input", {}).get("prompt")

        if not user_message:
            raise HTTPException(status_code=400, detail="No prompt found")

        result = agent.ask(user_message)
        return {
            "output": {
                "message": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": "strands-agent"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")

@app.get("/ping")
async def ping():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)