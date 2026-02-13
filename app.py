import boto3
import streamlit as st
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="DH-VLM Chat", layout="centered")
st.title("DH-VLM: Dior Homme Assistant")

client = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=os.getenv("AWS_REGION")
)

agent_id = os.getenv("AGENT_ID")
alias_id = os.getenv("ALIAS_ID")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about Dior Homme AW04..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            response = client.invoke_agent(
                agentId=agent_id,
                agentAliasId=alias_id,
                sessionId=st.session_state.session_id,
                inputText=prompt,
            )

            for event in response.get("completion"):
                if 'chunk' in event:
                    chunk_text = event["chunk"]["bytes"].decode()
                    full_response += chunk_text
                    response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Error: {str(e)}")