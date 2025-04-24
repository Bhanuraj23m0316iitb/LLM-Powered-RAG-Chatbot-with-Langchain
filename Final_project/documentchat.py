import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import LLMChain
from langchain_community.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
import io
import requests
import re
from dotenv import load_dotenv
import os
load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)

system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

The answer is foo
SOURCES: xyz

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

embeddings_model = CohereEmbeddings(cohere_api_key="COHERE_API_KEY")

def initialize_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")
    search = SerpAPIWrapper()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for answering questions about current events or questions that require logical analysis. Ask targeted questions for best results."
        )
    ]

    prefix = """You are Assistant. A helpful, friendly, informative, and intelligent chatbot, created by Pranav , who studies at IITB. Always provide detailed yet concise answers. You have access to the following tools if absolutely necessary:"""
    suffix = """Begin!

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    agent_prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=20)
    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

    return agent_chain, llm

agent_chain, llm = initialize_agent()

st.title("PDF Chatbot")
st.text("Ask something!")

uploaded_files = st.file_uploader("Upload PDF/Text files", accept_multiple_files=True, type=["pdf", "txt"])
user_input = st.text_input("Your message:")
submit_button = st.button("Submit")

def process_files(uploaded_files):
    all_texts, all_metadatas = [], []

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1]

        if file_extension == 'pdf':
            pdf_reader = PdfReader(uploaded_file)
            pdf_text = "".join(page.extract_text() for page in pdf_reader.pages)
            texts_split = text_splitter.split_text(pdf_text)
            metadatas = [{"source": f"{uploaded_file.name}-{i}"} for i in range(len(texts_split))]
            all_texts.extend(texts_split)
            all_metadatas.extend(metadatas)

        elif file_extension == 'txt':
            text_content = uploaded_file.read().decode().strip()
            texts_split = text_splitter.split_text(text_content)
            metadatas = [{"source": f"{uploaded_file.name}-{i}"} for i in range(len(texts_split))]
            all_texts.extend(texts_split)
            all_metadatas.extend(metadatas)

    if all_texts:
        embeddings = embeddings_model
        docsearch = Chroma.from_texts(all_texts, embeddings, metadatas=all_metadatas)
        return docsearch, all_metadatas, all_texts
    return None, None, None

if submit_button and user_input:
    if uploaded_files:
        docsearch, all_metadatas, all_texts = process_files(uploaded_files)
        if docsearch:
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm,
                chain_type="stuff",
                retriever=docsearch.as_retriever(),
            )
            res = chain({"input": user_input})
            answer = res["answer"]
            sources = res["sources"]
            st.write(f"{answer}\n\nSOURCES: {sources}")
        else:
            st.write("Unsupported file format. Please provide a PDF or text file.")
    else:
        res = agent_chain({"input": user_input})
        st.write(res["output"])
