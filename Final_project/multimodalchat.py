import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import LLMChain
from langchain_community.utilities import SerpAPIWrapper
from langchain import LLMMathChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
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
from PIL import Image
import tempfile
import os
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from dotenv import load_dotenv
import shutil
load_dotenv()

COHERE_API_KEY = os.getenv('COHERE_API_KEY')

def image2base64(image_data):
    encoded_string = base64.b64encode(image_data).decode("utf-8")
    return encoded_string

from pydub import AudioSegment

def transcribe_audio(file_path):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        # Convert to WAV format
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            wav_path = tmp_wav.name

        # Now use speech_recognition to transcribe the WAV file
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)

        # Clean up the temporary WAV file
        os.remove(wav_path)

        return transcription

    except Exception as e:
        return f"Error during transcription: {e}"


def extract_frames(video_path, frame_rate=1):
    with tempfile.TemporaryDirectory() as tmpdir:
        with VideoFileClip(video_path) as clip:
            duration = int(clip.duration)
            frames = []
            for t in range(0, duration * frame_rate, frame_rate):
                frame = clip.get_frame(t / frame_rate)
                frame_path = os.path.join(tmpdir, f"frame_{t}.png")
                img = Image.fromarray(frame)
                img.save(frame_path)
                with open(frame_path, "rb") as img_file:
                    frames.append(img_file.read())
    return frames

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

embeddings_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

def create_agent_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")

    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for answering questions about current events or questions that require logical analysis. Ask targeted questions for best results."
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.invoke,
            description="Handles mathematical queries and calculations."
        )
    ]

    prefix = """You are Agent. A helpful, friendly, informative, and intelligent chatbot, created by Pranav, who studies at IITB . Always provide detailed yet concise answers. You have access to the following tools if absolutely necessary:"""
    suffix = """Begin!

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=20)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)
    return agent_chain, llm

st.set_page_config(page_title="Multimodal Chatbot", layout="wide")

st.title("Multimodal Chatbot")

agent_chain, llm = create_agent_chain()

uploaded_files = st.file_uploader("Upload a file", accept_multiple_files=True)
user_query = st.text_input("Ask me!")

if st.button("Submit") or user_query:
    response = ""
    memory = agent_chain.memory

    images = [file for file in uploaded_files if file.type.startswith("image")]
    pdfs = [file for file in uploaded_files if file.type.endswith("pdf")]
    texts = [file for file in uploaded_files if file.type.startswith("text")]
    audio_files = [file for file in uploaded_files if file.type.startswith("audio")]
    video_files = [file for file in uploaded_files if file.type.startswith("video")]

    if images:
        st.write("Uploading image...")
        base64_image = image2base64(images[0].getvalue())
        image_url = f"data:image/png;base64,{base64_image}"

        image_message = HumanMessage(
            content=[
                {"type": "text", "text": user_query if user_query else "What's in this image?"},
                {"type": "image_url", "image_url": image_url},
            ]
        )
        image_response = llm.invoke([image_message]).content
        st.image(images[0])
        st.write(image_response)
        response += image_response

        memory.save_context(
            {"content": user_query if user_query else "Image received"},
            {"content": image_response}
        )

    elif pdfs:
        file_path = pdfs[0].getvalue()

        st.write(f"Processing {pdfs[0].name}...")

        pdf_stream = io.BytesIO(file_path)
        pdf = PdfReader(pdf_stream)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        texts = text_splitter.split_text(pdf_text)
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        embeddings = embeddings_model
        docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
        )

        st.write(f"Processing {pdfs[0].name} done. You can now ask questions!")

        res = chain(user_query)

        answer = res["answer"]
        sources = res["sources"].strip()
        source_elements = []

        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]
        all_sources = [m["source"] for m in metadatas]
        texts = text_splitter.split_text(pdf_text)

        if sources:
            found_sources = []
            for source in sources.split(","):
                source_name = source.strip().replace(".", "")
                try:
                    index = all_sources.index(source_name)
                except ValueError:
                    continue
                text = texts[index]
                found_sources.append(source_name)
                source_elements.append(st.text(text))

            if found_sources:
                answer += f"\nSources: {', '.join(found_sources)}"
            else:
                answer += "\nNo sources found"

        memory.save_context(
            {"content": user_query},
            {"content": answer}
        )
        st.write(answer)

    elif audio_files:
        file_path = audio_files[0].getvalue()
        transcription_response = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file_path)
                tmp_path = tmp.name

            transcription = transcribe_audio(tmp_path)
            transcription_message = HumanMessage(content=f"The user has input an audio file and some program has converted it into transcription for you to use. The user does not know this and you don't have to mention this. \n\nAudio transcription: {transcription}.\n\n Use this information to answer the following query: {user_query}")
            transcription_response = llm.invoke([transcription_message]).content
            st.write(transcription_response)
        except Exception as e:
            st.write(f"Error processing audio file: {e}")

        memory.save_context(
            {"content": user_query},
            {"content": transcription_response}
        )

    elif video_files:
        video_path = video_files[0].getvalue()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_path)
                tmp_path = tmp.name

            st.write("Extracting frames from video...")
            frames = extract_frames(tmp_path)
            frame_urls = [f"data:image/png;base64,{image2base64(frame)}" for frame in frames]
            frame_message = HumanMessage(
                content=[
                    {"type": "text", "text": user_query},
                    *[{"type ": "image_url", "image_url": url} for url in frame_urls],
                ]
            )
            frame_response = llm.invoke([frame_message]).content
            st.write(frame_response)
        except Exception as e:
            st.write(f"Error processing video file: {e}")

        memory.save_context(
            {"content": user_query},
            {"content": frame_response}
        )
