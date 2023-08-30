from pydantic import BaseModel
from typing import List, Optional
from decouple import config

DEFAULT_MODEL_NAME = config("DEFAULT_MODEL_NAME", default="llama2_7b_chat_uncensored", cast=str)
DEFAULT_OBJECT_NAME = "embedding"

# Request/Response models start here:
class EmbeddingRequest(BaseModel):
    input: str
    model: Optional[str] = DEFAULT_MODEL_NAME

class EmbeddingResponse(BaseModel):
    index: int
    object: str = DEFAULT_OBJECT_NAME
    embedding: List[float]
