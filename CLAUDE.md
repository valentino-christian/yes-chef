# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**yes-chef** is a Python-based recipe management system that:
- Extracts recipes from Google Docs using the Google Docs API
- Summarizes recipe URLs using web scraping (Selenium) and OpenAI GPT-4o-mini
- Stores recipe embeddings in ChromaDB for semantic search
- Uses HuggingFace embeddings (Qwen/Qwen3-Embedding-8B) for vector representations

## Architecture

### Core Components

1. **GoogleDocRecipeParser** (`get_recipes.py`)
   - Main orchestrator for recipe extraction and embedding
   - Authenticates with Google Docs API using OAuth2
   - Extracts recipes from structured Google Docs (HEADING_2 as recipe titles)
   - Detects recipe URLs in doc content and triggers web summarization
   - Embeds recipes using HuggingFace models and stores in ChromaDB

2. **Recipe URL Summarizer** (`webpage_recipe_summarizer.py`)
   - Uses Selenium WebDriver to scrape recipe websites (handles JavaScript-rendered content)
   - Extracts and cleans HTML content using BeautifulSoup
   - Summarizes recipes using OpenAI GPT-4o-mini
   - Returns structured recipe summaries with ingredients and instructions

3. **Database Layer**
   - ChromaDB persistent storage at `recipes_db/chroma.sqlite3`
   - Collection: "recipe_text"
   - Each recipe stored with: name, full content, and embedding vector

### Data Flow

```
Google Doc → GoogleDocRecipeParser.extract_recipes_from_google_doc()
    ↓
Parse HEADING_2 sections as individual recipes
    ↓
Detect URLs in recipe content → webpage_recipe_summarizer.summarize_recipe_url()
    ↓
Generate embeddings via HuggingFace API
    ↓
Store in ChromaDB (recipes_db/)
```

## Setup and Configuration

### Authentication Files

- `credentials.json` - Google OAuth client credentials (required for Google Docs API)
- `token.json` - Stores user's access/refresh tokens (auto-generated on first auth)

### Environment Variables

The project expects:
- `OPENAI_API_KEY` - OpenAI API key for recipe summarization
  - Currently hardcoded in `webpage_recipe_summarizer.py` and `connection_test.py`
  - Should be moved to `.env` file (project uses python-dotenv)

### ChromeDriver Configuration

- Path to ChromeDriver is hardcoded: `\Users\Val\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe`
- Update `PATH_TO_CHROME_DRIVER` in `webpage_recipe_summarizer.py` and `connection_test.py` if needed

## Running the Project

### Main Recipe Extraction

```bash
python get_recipes.py
```

This will:
1. Authenticate with Google Docs (opens browser for OAuth if needed)
2. Extract recipes from the hardcoded Google Doc ID: `1KAHavgSWbBiOtcxFqFvp4l79MITUoxsshruc5am_pK0`
3. Summarize any recipe URLs found in the content
4. Generate embeddings and store in ChromaDB

### Test Scripts

```bash
# Test Google Docs API connection
python doc_test.py

# Test webpage summarization
python connection_test.py
```

## Key Dependencies

- `google-auth`, `google-auth-oauthlib`, `google-api-python-client` - Google Docs API
- `chromadb` - Vector database
- `huggingface-hub` - Embedding generation
- `openai` - Recipe summarization
- `selenium` - Web scraping
- `beautifulsoup4` - HTML parsing
- `pandas` - Data manipulation

## Important Notes

### Security Concerns

- **API keys and tokens are currently hardcoded or committed** (`credentials.json`, `token.json`)
- HuggingFace token is hardcoded in `get_recipes.py:20`
- OpenAI API key is hardcoded with fallback in `webpage_recipe_summarizer.py:14`
- These should be moved to environment variables and added to `.gitignore`

### Recipe Parsing Logic

- Recipes are identified by `HEADING_2` style in Google Docs
- Content under each heading until the next `HEADING_2` is considered part of that recipe
- Short content (≤4 lines) containing URLs triggers automatic web summarization
- Web summaries are appended to recipe content under "--- Web Summary ---"

### ChromaDB Collection

- Collection name: `"recipe_text"`
- Documents format: `"{recipe_name}\n{recipe_content}"`
- IDs: Recipe names (must be unique)
- Embeddings: 8192-dimensional vectors from Qwen/Qwen3-Embedding-8B

## Modifying the Code

### To change the Google Doc source:
Update `document_id` in `get_recipes.py:223`

### To change embedding model:
Update `model` parameter in `get_recipes.py:170`

### To change summarization model:
Update `model` in `webpage_recipe_summarizer.py:90`

### To recreate ChromaDB:
Delete `recipes_db/` directory before running `get_recipes.py` (ChromaDB will auto-create on `create_collection`)
