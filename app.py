import os
import traceback
from typing import List, Optional
from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from huggingface_hub import InferenceClient
import chromadb
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import HTTPException

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class RecipeQuery(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = []

# Custom embedding function using HuggingFace Inference API (same as get_recipes.py)
class HFInferenceEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.client = InferenceClient(token=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.client.feature_extraction(texts, model=self.model)
        return [vec.tolist() if hasattr(vec, 'tolist') else list(vec) for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


vectordb = None
qa_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectordb, qa_chain
    print("🚀 Starting application...")

    load_dotenv()
    HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    if not HF_TOKEN:
        raise ValueError("HUGGINGFACE_API_TOKEN environment variable not set!")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set!")

    print("✅ Environment variables loaded")

    # ChromaDB bundled in Docker image (read-only)
    print("📂 Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./recipes_db")

    # Use HuggingFace Inference API (same as used to create embeddings)
    print("🤗 Setting up HuggingFace embeddings...")
    embedding_function = HFInferenceEmbeddings(
        api_key=HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("🔍 Loading vector database...")
    vectordb = Chroma(
        client=chroma_client,
        collection_name="recipe_text",
        embedding_function=embedding_function
    )

    print("🧠 Initializing LLM...")
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=512
    )

    prompt_template = """You are a helpful recipe assistant. Use the following recipe information to answer the question.

IMPORTANT: If any of the context below contains error messages, stack traces, ChromeDriver errors, or technical debugging information, completely ignore that content. Only use actual recipe information (ingredients, instructions, cooking tips).

When the user mentions ingredients they have available (e.g., "I have tomatoes and chicken"), look through the recipes in the context and suggest ones that use those ingredients. Explain which recipe(s) would work and why, and offer to provide the user with the recipe.

When the user asks what they can make, recommend recipes from the context that best match their available ingredients or preferences.

If none of the recipes in the context are relevant to the user's question or ingredients, say "I don't have a recipe that matches, but here's what I found:" and briefly describe the available recipes.

Context:
{context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    print("⛓️ Creating QA chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    print("✅ Application ready!")
    yield  # App runs between startup and shutdown

    print("🛑 Shutting down...")

    vectordb = None
    qa_chain = None

app = FastAPI(lifespan=lifespan)

@app.post("/getRecipeRecommendation")
async def get_recipe_recommendation(request: RecipeQuery) -> dict:
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        print(f"📥 Received query: {request.query}")

        # Convert chat history to list of tuples (human, ai)
        history_tuples = []
        for i in range(0, len(request.chat_history) - 1, 2):
            if request.chat_history[i].role == "user" and i + 1 < len(request.chat_history):
                history_tuples.append((
                    request.chat_history[i].content,
                    request.chat_history[i + 1].content
                ))

        response = qa_chain.invoke({
            "question": request.query,
            "chat_history": history_tuples
        })
        print(f"✅ Got response: {response}")
        return {"result": response["answer"]}
    except Exception as e:
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"❌ Error occurred: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {type(e).__name__}: {str(e) if str(e) else repr(e)}"
        )

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)