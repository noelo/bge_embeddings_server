from pydantic import BaseModel
from typing import List, Optional

DEFAULT_MODEL_NAME = "ggml-model-q4_0"
DEFAULT_OBJECT_NAME = "embedding"

# Request/Response models start here:
class EmbeddingRequest(BaseModel):
    input: str
    model: Optional[str] = DEFAULT_MODEL_NAME


class EmbeddingResponse(BaseModel):
    index: int
    object: str = DEFAULT_OBJECT_NAME
    embedding: List[float]
