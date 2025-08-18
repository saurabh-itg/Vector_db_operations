from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Ensure your GROQ_API_KEY is set in your .env file
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set!")

# Initialize the model
# Using a model known for translation tasks
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# 1. Create a clear and reusable prompt template
system_template = "Translate the following text into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Define the output parser
parser = StrOutputParser()

# 3. Construct the chain
chain = prompt_template | model | parser

# 4. Define the Pydantic model for the request body
# This provides clear validation and schema for your API endpoint.
class TranslationRequest(BaseModel):
    """Input model for the translation request."""
    language: str
    text: str

# 5. Initialize the FastAPI application
app = FastAPI(
    title="LangChain Translation Server",
    version="1.0",
    description="A simple API server for translations using LangChain and Groq."
)

# 6. Define a standard FastAPI endpoint instead of using add_routes
# This gives us full control and avoids the Pydantic schema generation conflict.
@app.post("/translate")
async def translate_text(request: TranslationRequest):
    """
    Receives a translation request and returns the translated text.
    """
    # The chain expects a dictionary as input that matches the prompt template variables
    input_data = {"language": request.language, "text": request.text}
    
    # Use ainvoke for asynchronous execution with FastAPI
    result = await chain.ainvoke(input_data)
    
    return {"translation": result}

# 7. Add a root path for basic health check or info
@app.get("/")
def read_root():
    return {"message": "Welcome to the Translation API! Visit /docs for details."}


# 8. Run the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    # The app is run on localhost port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
