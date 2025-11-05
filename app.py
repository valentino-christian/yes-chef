import os
import traceback
from typing import List
from fastapi import FastAPI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from huggingface_hub import InferenceClient
import chromadb
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import HTTPException

class RecipeQuery(BaseModel):
    query: str

# Custom embedding function using HuggingFace Inference API (same as get_recipes.py)
class HFInferenceEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "Qwen/Qwen3-Embedding-8B"):
        self.client = InferenceClient(provider="auto", api_key=api_key)
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

    if not HF_TOKEN:
        raise ValueError("HUGGINGFACE_API_TOKEN environment variable not set!")

    print("✅ Environment variables loaded")

    # ChromaDB bundled in Docker image (read-only)
    print("📂 Initializing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./recipes_db")

    # Use HuggingFace Inference API (same as used to create embeddings)
    print("🤗 Setting up HuggingFace embeddings...")
    embedding_function = HFInferenceEmbeddings(
        api_key=HF_TOKEN,
        model="Qwen/Qwen3-Embedding-8B"
    )

    print("🔍 Loading vector database...")
    vectordb = Chroma(
        client=chroma_client,
        collection_name="recipe_text",
        embedding_function=embedding_function
    )

    print("🧠 Initializing LLM...")
    llm = HuggingFaceEndpoint(
        endpoint_url=f"https://api-inference.huggingface.co/models/Qwen/Qwen3-4B-Instruct-2507-FP8",
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 512,
        }
    )

    print("⛓️ Creating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
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
        response = qa_chain.invoke(request.query)
        print(f"✅ Got response: {response}")
        return {"result": response["result"]}
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