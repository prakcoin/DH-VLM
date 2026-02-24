import boto3
import json
import base64
import os
import pandas as pd
from collections import defaultdict

MODEL_ID = "amazon.nova-pro-v1:0"
IMAGE_DIR = "./images"
OUTPUT_CSV = "pieces.csv"

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_look_groups(directory):
    groups = defaultdict(list)
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            look_id = filename.split('_')[0]
            groups[look_id].append(os.path.join(directory, filename))
    return groups

def analyze_look(look_id, image_paths):
    content = []
    for path in image_paths:
        content.append({
            "image": {
                "format": "jpeg",
                "bytes": encode_image(path)
            }
        })
    
    prompt = f"""
    Analyze the runway look in these images for {look_id}. 
    Identify every distinct piece of clothing and accessory worn by the model. 
    
    IMPORTANT: Return a JSON list where EACH item (e.g., the jacket, the shirt, the trousers, the shoes, the belt) is its own separate object.
    
    Return a JSON list of objects with these exact keys:
    "Name": "",
    "Reference Code": "Not available",
    "Look Number": "{look_id}",
    "Category": (Must be one of: Accessories, Bottom, Footwear, Outerwear, Top),
    "Subcategory": (e.g., T-Shirt, Belt, Jeans),
    "Primary Color": (Dominant color),
    "Secondary Color(s)": (List secondary colors or "no secondary color"),
    "Pattern": (e.g., Solid, Plaid, Striped),
    "Primary Outer Material": "",
    "Secondary Outer Material(s)": "None",
    "Additional Notes": ""
    """
    
    content.append({"type": "text", "text": prompt})

    body = json.dumps({
        "inferenceConfig": {
            "max_new_tokens": 700,
        },
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    })

    response = bedrock.invoke_model(
        modelId=MODEL_ID, 
        body=body
    )

    response_body = json.loads(response.get("body").read())
    
    try:
        return json.loads(response_body["output"]["message"]['content'][0]['text'])
    except:
        print(f"Error parsing JSON for {look_id}")
        return []

all_rows = []
groups = get_look_groups(IMAGE_DIR)

for look_id, paths in groups.items():
    print(f"Processing {look_id}...")
    items = analyze_look(look_id, paths)
    for item in items:
        item["Images"] = ", ".join([os.path.basename(p) for p in paths])
        all_rows.append(item)

df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Done. Data saved to {OUTPUT_CSV}")