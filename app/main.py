"""
The Semantic Bridge - FastAPI Application

A semantic translation system from English to Arabic using AMR.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Find .env file (check current dir and parent dir)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded .env from {env_path}")
else:
    load_dotenv()  # Try default locations

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import SemanticBridge, TranslationStatus


# Global pipeline instance
pipeline: Optional[SemanticBridge] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global pipeline
    
    # Determine if we should use mock mode
    use_mock = os.getenv("USE_MOCK", "true").lower() == "true"
    llm_provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL")
    
    # Check if API keys are available
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_gemini_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    logger.info(f"API Keys loaded - OpenAI: {has_openai_key}, Anthropic: {has_anthropic_key}, Gemini: {has_gemini_key}")
    
    logger.info(f"Initializing Semantic Bridge (mock={use_mock}, provider={llm_provider})")
    
    # Lower threshold and more retries for better complex sentence handling
    verification_threshold = float(os.getenv("VERIFICATION_THRESHOLD", "0.65"))
    max_retries = int(os.getenv("MAX_RETRIES", "4"))
    
    pipeline = SemanticBridge(
        llm_provider=llm_provider,
        llm_model=llm_model,
        use_mock=use_mock,
        verification_threshold=verification_threshold,
        max_retries=max_retries
    )
    
    logger.info(f"Verification threshold: {verification_threshold}, Max retries: {max_retries}")
    
    logger.info("Semantic Bridge initialized successfully")
    yield
    
    logger.info("Shutting down Semantic Bridge")


# Create FastAPI app
app = FastAPI(
    title="The Semantic Bridge",
    description="English to Arabic Semantic Translation using AMR",
    version="0.1.0",
    lifespan=lifespan
)

# Setup templates
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Setup static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class TranslationRequest(BaseModel):
    """Request model for translation."""
    text: str
    provider: Optional[str] = None  # openai, anthropic, gemini
    model: Optional[str] = None     # Specific model name
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "The committee did not approve the decision.",
                "provider": "openai",
                "model": "gpt-4-turbo-preview"
            }
        }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/translate")
async def translate(request: TranslationRequest):
    """
    Translate English text to Arabic with semantic verification.
    
    The process:
    1. Parse English → AMR (PropBank-based semantic graph)
    2. Render AMR → Arabic (LLM with constraints)
    3. Parse Arabic → AMR (LLM-based reverse parsing)
    4. Compare AMR graphs (Smatch metric)
    
    Returns the full pipeline result including step-by-step details.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Use provided provider/model or fall back to defaults
        provider = request.provider or os.getenv("LLM_PROVIDER", "openai")
        model = request.model
        
        logger.info(f"Translation request with provider={provider}, model={model or 'default'}")
        
        result = await pipeline.translate_async(
            request.text,
            provider_override=provider,
            model_override=model
        )
        # Return full dict including steps array
        return JSONResponse(content=result.to_dict())
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None
    }


@app.get("/providers")
async def get_providers():
    """Get available LLM providers and their models."""
    providers = {
        "openai": {
            "name": "OpenAI",
            "available": bool(os.getenv("OPENAI_API_KEY")),
            "models": [
                {"id": "gpt-4-turbo-preview", "name": "GPT-4 Turbo"},
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
            ]
        },
        "anthropic": {
            "name": "Anthropic",
            "available": bool(os.getenv("ANTHROPIC_API_KEY")),
            "models": [
                {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus"},
                {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet"},
                {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
                {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"}
            ]
        },
        "gemini": {
            "name": "Google Gemini",
            "available": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "models": [
                {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash"},
                {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
                {"id": "gemini-3-pro-preview", "name": "Gemini 3 Pro Preview"},
                {"id": "gemini-3-flash-preview", "name": "Gemini 3 Flash Preview"}
            ]
        }
    }
    
    return {
        "providers": providers,
        "default_provider": os.getenv("LLM_PROVIDER", "openai")
    }


@app.get("/examples")
async def examples():
    """Get example sentences for testing."""
    return {
        "examples": [
            {
                "text": "The committee did not approve the decision.",
                "description": "Tests negation preservation"
            },
            {
                "text": "The boy wants to go to school.",
                "description": "Tests embedded clause structure"
            },
            {
                "text": "She believes that he is honest.",
                "description": "Tests propositional attitude"
            },
            {
                "text": "The teacher gave the student a book.",
                "description": "Tests ditransitive argument structure"
            },
            {
                "text": "If it rains, the match will be cancelled.",
                "description": "Tests conditional structure"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug
    )

