# DH-Agent: Dior Homme Archive Assistant
This project implements an AI fashion agent through AWS Bedrock specializing in the Dior Homme archive. Powered by the Nova Pro foundation model, the agent reasons over a multimodal knowledge base of runway look images and structured look breakdown data. For accurate answer synthesis, it uses four toolsets: ItemTools for single item analysis, AggregationTools for collection-wide analysis, ImageTools for visual analysis, and SearchTools for real-time market listings and out-of-scope information.

## Setup
1. Create a virtual environment:
   `python -m venv .venv`
2. Activate it:
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`
3. To install the packages run:
    `pip install -r requirements.txt`
4. Enviornment Variables
    Create a `.env` file in the root directory and add your AWS and Label Studio credentials.

## Usage
To start the app run:
```
streamlit run app.py
```