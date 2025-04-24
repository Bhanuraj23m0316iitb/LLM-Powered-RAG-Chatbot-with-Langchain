import streamlit as st
from streamlit import runtime
import subprocess
import os

st.set_page_config(page_title="Main App", layout="wide")

st.title("Main App")

app_options = {
    "Multimodal Chat": "multimodalchat.py",
    "Insti GPT": "instigpt1.py",
    "Document Chatbot": "documentchat.py",
}

app_choice = st.selectbox("Select an App to run", list(app_options.keys()))

if st.button("Run App"):
    if app_choice:
        selected_app = app_options[app_choice]
        subprocess.Popen(["streamlit", "run", selected_app])

st.write("Select an app from the dropdown above and click 'Run App' to start.")
