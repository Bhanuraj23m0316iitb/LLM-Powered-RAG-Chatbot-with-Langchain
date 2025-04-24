import streamlit as st
from LLMs import *
import time

# Streamlit app configuration
st.set_page_config(page_title="LLM Arena", layout="wide")

# App title
st.title("LLM Arena")

# Prompt input
prompt = st.text_area("Enter your prompt here:")

# Store responses and likes in session state
if 'responses' not in st.session_state:
    st.session_state.responses = {}
    st.session_state.likes = {}

# Function to display response with like button
def display_response(model_name, response, time_taken):
    if model_name not in st.session_state.responses:
        st.session_state.responses[model_name] = []
        st.session_state.likes[model_name] = []

    st.session_state.responses[model_name].append(response)
    st.session_state.likes[model_name].append(0)

    st.write(f"### {model_name} Response")
    st.write(response)
    st.write(f"*Response Time: {time_taken:.2f} seconds*")
    
    if st.button(f"Like {model_name} Response {len(st.session_state.responses[model_name])}", key=f"like_{model_name}_{len(st.session_state.responses[model_name])}"):
        st.session_state.likes[model_name][-1] += 1

    st.write(f"Likes: {st.session_state.likes[model_name][-1]}")

# Function to clear all responses
def clear_responses():
    st.session_state.responses = {}
    st.session_state.likes = {}

# Buttons for each LLM
if st.button("Get Llama70B Response"):
    start_time = time.time()
    response = get_llama70b_response(prompt)
    end_time = time.time()
    display_response("Llama70B", response, end_time - start_time)

if st.button("Get Mistral Response"):
    start_time = time.time()
    response = get_mistral_response(prompt)
    end_time = time.time()
    display_response("Mistral", response, end_time - start_time)

if st.button("Get Llama3 Response"):
    start_time = time.time()
    response = get_llama3_8b_response(prompt)
    end_time = time.time()
    display_response("Llama3", response, end_time - start_time)

if st.button("Get Claude Response"):
    start_time = time.time()
    response = get_claude_response(prompt)
    end_time = time.time()
    display_response("Claude", response, end_time - start_time)

if st.button("Get Gemma Response"):
    start_time = time.time()
    response = get_Gemma_response(prompt)
    end_time = time.time()
    display_response("Gemma", response, end_time - start_time)

if st.button("Get Gemini Response"):
    start_time = time.time()
    response = get_Gemini_response(prompt)
    end_time = time.time()
    display_response("Gemini", response, end_time - start_time)

# Clear all responses button
if st.button("Clear All Responses"):
    clear_responses()
    st.write("All responses cleared.")

# Display response history
st.write("## Response History")
for model_name, responses in st.session_state.responses.items():
    st.write(f"### {model_name} Response History")
    for i, response in enumerate(responses):
        st.write(f"**Response {i+1}:**")
        st.write(response)
        st.write(f"Likes: {st.session_state.likes[model_name][i]}")
