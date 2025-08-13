import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Ollama Model and LangChain Setup
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()

# 2. Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question:{question}")
    ]
)

# 3. Create a chain combining prompt, model, and output parser
chain = prompt | llm | output_parser

# 4. Streamlit UI
st.title("Gemma 2B Chatbot powered by Ollama and LangChain")
input_text = st.text_input("Ask a question to Gemma 2B:")

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
