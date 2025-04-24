import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import google.generativeai as genai
from groq import Groq
import anthropic


# Load environment variables
load_dotenv()

# Load API keys from environment variables

key = os.getenv('HUGGINGFACE_API_KEY')
gemini = os.getenv('GOOGLE_API_KEY')
groq_api = os.getenv('GROQ_API_KEY')
claude_api=os.getenv('CLAUDE_API_KEY')



def get_mistral_response(prompt):
    client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content":prompt,
        }
    ],
    model="mixtral-8x7b-32768",
    )

    return chat_completion.choices[0].message.content

def get_llama70b_response(prompt):
    client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content":prompt,
        }
    ],
    model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content


def get_claude_response(prompt):
   

    client = anthropic.Anthropic(api_key=claude_api)

    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.4,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return message.content



def get_llama3_8b_response(prompt):
    client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content":prompt,
        }
    ],
    model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content



def get_Gemma_response(prompt):
    client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    )

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content":prompt,
        }
    ],
    model="gemma2-9b-it",
    )
    return chat_completion.choices[0].message.content

def get_Gemini_response(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    return response.text



