import streamlit as st
import time
import argparse

import os
import warnings
from pipeline.agent import Agent

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("path", default = None, type=str, help="Define Path to your's repository if you want to use context from it")
args = parser.parse_args()

@st.cache_resource
def define_agent():
    return Agent(args.path)

# Streamed response emulator
def response_generator(agent, prompt):
    response = agent(prompt)
    for word in response:
        yield word
        time.sleep(0.02)

#st.title("AI coding assistant with RAG")
agent = define_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can i help You?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner('Generating...'):
            response = st.write_stream(response_generator(agent, prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})