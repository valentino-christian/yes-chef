import os
from fastapi import FastAPI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pydantic import BaseModel
from fastapi import HTTPException

class RecipeQuery(BaseModel):
    query: str

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
    embedding_function = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="Qwen/Qwen3-Embedding-8B"
    )

    print("🔍 Loading vector database...")
    vectordb = Chroma(
        client=chroma_client,
        collection_name="recipe_text",
        embedding_function=embedding_function
    )

    print("🧠 Initializing LLM...")
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-4B-Instruct-2507-FP8",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.7,
        max_new_tokens=512,
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
        response = qa_chain.invoke(request.query)
        return {"result": response["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)