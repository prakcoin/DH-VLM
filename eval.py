import boto3
import time
import os
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION")
KB_ID = os.getenv("KB_ID")
ROLE_ARN = os.getenv("ROLE_ARN")
DATASET_S3_URI = os.getenv("DATASET_S3_URI")
OUTPUT_S3_URI = os.getenv("OUTPUT_S3_URI")

client = boto3.client("bedrock", region_name=REGION)

def create_eval_job():
    job_response = client.create_evaluation_job(
        jobName="aw04-rag-eval",
        jobDescription="Retrieve-and-generate evaluation for aw04-agent",
        applicationType="RagEvaluation",
        roleArn=ROLE_ARN,

        inferenceConfig={
            "ragConfigs": [
                {
                    "knowledgeBaseConfig": {
                        "retrieveAndGenerateConfig": {
                            "type": "KNOWLEDGE_BASE",
                            "knowledgeBaseConfiguration": {
                                "knowledgeBaseId": KB_ID,
                                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0",

                                "retrievalConfiguration": {
                                    "vectorSearchConfiguration": {
                                        "numberOfResults": 10,
                                    }
                                }
                            }
                        }
                    }
                }
            ]
        },

        outputDataConfig={
            "s3Uri": OUTPUT_S3_URI
        },

        evaluationConfig={
            "automated": {
                "datasetMetricConfigs": [
                    {
                        "taskType": "Generation",
                        "dataset": {
                            "name": "RagDataset",
                            "datasetLocation": {
                                "s3Uri": DATASET_S3_URI
                            }
                        },
                        "metricNames": [
                            "Builtin.Faithfulness",
                            "Builtin.Correctness",
                            "Builtin.Completeness",
                            "Builtin.CitationPrecision",
                            "Builtin.Refusal"
                        ]
                    }
                ],
                "evaluatorModelConfig": {
                    "bedrockEvaluatorModels": [
                        {
                            "modelIdentifier": "amazon.nova-pro-v1:0"
                        }
                    ]
                }
            }
        }
    )

    return job_response

def wait_for_completion(job_response):
    while True:
        job = client.get_evaluation_job(jobIdentifier=job_response['jobArn'])
        status = job["status"] 

        if status in ["Completed", "Failed", "Stopped"]:
            return job["outputDataConfig"]

        time.sleep(30)

if __name__ == "__main__":
    job_arn = create_eval_job()
    print("Started evaluation job:", job_arn)

    final_job = wait_for_completion(job_arn)
    print("Result URI:", final_job["s3Uri"])
