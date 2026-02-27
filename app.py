import streamlit as st
import uuid
from src.agent import DHAgent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="DH-VLM Chat", layout="centered")
st.title("DH-VLM: Dior Homme Assistant")

if "agent" not in st.session_state:
    st.session_state.agent = DHAgent()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about Dior Homme Autumn/Winter 2004..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            for chunk in st.session_state.agent.invoke(prompt, st.session_state.session_id):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")