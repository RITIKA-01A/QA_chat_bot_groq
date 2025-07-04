from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
import os

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

## Langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ["LANGCHAIN_PROJECT"]='QA chatbot with groq'

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a helpful assistant.Please response to the user queries"),
        ("user","Question:{question}")
    ]

)

output_parser = StrOutputParser()
def generate_response(question,model,temperature,max_tokens):

    llm = ChatGroq(model=model,temperature=temperature,max_tokens=max_tokens)  
    chain=prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer


## Title of the app
st.title("Enhanced Q&A Chatbot With Groq")


## Dropdown to select the various Open ai models
model = st.sidebar.selectbox("Select an Groq AI Model",['llama-3.3-70b-versatile','llama3-70b-8192'])

## Adjust response parameter
temperature = st.sidebar.slider("Temparature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

## Main interface for the user
st.write("Go ahead and ask any question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,model,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")