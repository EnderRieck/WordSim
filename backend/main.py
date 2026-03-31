import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import ModelNotFoundError, ModelNotReadyError, SimilarityModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SentencePair(BaseModel):
    sentence1: str
    sentence2: str
    model_id: str | None = None


model_service = None


@app.on_event("startup")
async def load_default_model():
    global model_service
    model_service = SimilarityModel.get_instance()
    model_service.warmup_default_model()


@app.get("/models")
async def get_models():
    return {"models": model_service.list_models(), "default_model_id": model_service.default_model_id}


@app.post("/similarity")
async def calculate_similarity(pair: SentencePair):
    start = time.time()
    try:
        result = model_service.predict_similarity(pair.sentence1, pair.sentence2, pair.model_id)
    except ModelNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {error.args[0]}") from error
    except ModelNotReadyError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    result["time_ms"] = round((time.time() - start) * 1000, 2)
    return result


@app.post("/similarity/detailed")
async def calculate_detailed(pair: SentencePair):
    start = time.time()
    try:
        result = model_service.predict_with_attention(pair.sentence1, pair.sentence2, pair.model_id)
    except ModelNotFoundError as error:
        raise HTTPException(status_code=404, detail=f"Unknown model_id: {error.args[0]}") from error
    except ModelNotReadyError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    result["time_ms"] = round((time.time() - start) * 1000, 2)
    return result


frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="static")
