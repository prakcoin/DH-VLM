import boto3
import json
import base64
import os
import re
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "amazon.nova-pro-v1:0"
IMAGE_DIR = "./images"
OUTPUT_DIR = "./labels"

bedrock = boto3.client(service_name="bedrock-runtime", 
                       region_name="us-east-1",
                       aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                       aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_look_groups(directory):
    groups = defaultdict(list)
    
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            match = re.search(r'look(\d+)_(\d+)', filename)
            if match:
                look_id = int(match.group(1))
                seq_id = int(match.group(2))
                full_path = os.path.join(directory, filename)
                
                groups[look_id].append((seq_id, full_path))

    for look_id in groups:
        groups[look_id].sort()
        groups[look_id] = [path for _, path in groups[look_id]]

    return dict(sorted(groups.items()))

def analyze_look(look_id, image_paths):
    content = []
    for path in image_paths:
        content.append({
            "image": {
                "format": "jpeg",
                "source": {
                    "bytes": encode_image(path)
                }
            }
        })
    
    prompt = f"""
    Analyze the runway look in these images for {look_id}. 
    Identify every distinct piece of clothing and accessory worn by the model visible within the look images. 
    Return a single JSON list of objects. Each distinct piece of clothing or accessory worn by the model (e.g., the jacket, shirt, trousers, shoes, belt) must be its own separate object in the list.

    Each object must use these exact keys:
    "Name": (A concise, distinct name. Priority: [Material (Only include if it is a defining characteristic or unconventional for the item.)] [Misc Identifier] [Item Type]. Examples: "Leather Studded Belt", "Eye Graphic T-Shirt", "Rust Denim".),
    "Reference Code": "Not available",
    "Category": (Must be one of: Accessories, Bottom, Footwear, Outerwear, Top),
    "Subcategory": (e.g., T-Shirt, Belt, Jeans),
    "Primary Color": (Dominant color),
    "Secondary Color(s)": (List secondary colors or "No secondary color"),
    "Pattern": (e.g., Solid, Plaid, Striped),
    "Primary Outer Material": "None",
    "Secondary Outer Material(s)": "None",
    "Additional Notes": (Describe only distinguishing physical features like hardware, unique textures, specific fit, or distressing. If the item is a basic staple with no unique features, leave this as "Standard fit/design". Avoid subjective praise. Max 20 words.)
    """
    
    content.append({"text": prompt})

    body = json.dumps({
        "inferenceConfig": {
            "max_new_tokens": 2000,
            "temperature": 0.0
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
    print(response_body["output"]["message"]['content'][0]['text'])
    try:
        return json.loads(response_body["output"]["message"]['content'][0]['text'])
    except Exception as e:
        print(f"Error parsing JSON for {look_id}:", e)
        return []
    
os.makedirs(OUTPUT_DIR, exist_ok=True)

groups = get_look_groups(IMAGE_DIR)

for look_id, paths in list(groups.items())[:3]:
    print(f"Processing Look {look_id}...")
    items = analyze_look(look_id, paths)
    
    if items:
        image_list_str = ", ".join([os.path.basename(p) for p in paths])
        
        for item in items:
            item["Look Number"] = look_id
            item["Images"] = image_list_str
        
        cols = ["Name", "Reference Code", "Look Number", "Category", "Subcategory", 
                "Primary Color", "Secondary Color(s)", "Pattern", "Primary Outer Material", 
                "Secondary Outer Material(s)", "Additional Notes", "Images"]
        
        df_look = pd.DataFrame(items)
        
        df_look = df_look[[c for c in cols if c in df_look.columns]]
        
        file_path = os.path.join(OUTPUT_DIR, f"look{look_id}.csv")
        df_look.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")

print("Initial data creation complete.")