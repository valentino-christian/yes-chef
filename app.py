import os
from fastapi import FastAPI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
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
    load_dotenv()
    HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

    # ChromaDB bundled in Docker image (read-only)
    chroma_client = chromadb.PersistentClient(path="./recipes_db")
    embedding_function = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-8B")

    vectordb = Chroma(
        client=chroma_client,
        collection_name="recipe_text",
        embedding_function=embedding_function
    )
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-4B-Instruct-2507-FP8",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.7,
        max_new_tokens=512,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    yield  # App runs between startup and shutdown

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