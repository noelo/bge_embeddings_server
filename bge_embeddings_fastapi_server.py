import sys
sys.path.insert(0,'bert.cpp/python/')


from bertcpp import BertCppEmbeddings
from embeddings_models import EmbeddingResponse, EmbeddingRequest
from langchain.embeddings.base import Embeddings
import asyncio
import glob
from decouple import config
import json
import logging
import os 
import time
import traceback
import urllib.request
from collections import defaultdict
from datetime import datetime
from hashlib import sha3_256
from logging.handlers import RotatingFileHandler
from typing import List, Optional, Tuple, Dict
import numpy as np

# from decouple import config
import uvicorn
# import psutil
import fastapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, Response
from langchain.embeddings import LlamaCppEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
# from sqlalchemy import select
# from sqlalchemy import text as sql_text
# from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.orm import sessionmaker, joinedload
# import faiss
# import pandas as pd
# import PyPDF2
# from magic import Magic
# from llama_cpp import Llama
# from scipy.stats import rankdata
# from sklearn.preprocessing import KBinsDiscretizer
# from numba import jit
# from hyppo.independence import Hsic



logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger = logging.getLogger(__name__)
configured_logger = logger

# Global variables
DEFAULT_MODEL_TYPE = "BERT"
EMBEDDING_SERVER_LISTEN_PORT = config("LLAMA_EMBEDDING_SERVER_LISTEN_PORT", default=8089, cast=int)
DEFAULT_MODEL_NAME = config("DEFAULT_MODEL_NAME", default="ggml-model-q4_0.bin", cast=str)
DEFAULT_MODEL_DIR = config("DEFAULT_MODEL_DIR", default="/home/noelo/dev/bge_embeddings_server/models/", cast=str)
LLM_CONTEXT_SIZE_IN_TOKENS = config("LLM_CONTEXT_SIZE_IN_TOKENS", default=512, cast=int)
MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING = config("MINIMUM_STRING_LENGTH_FOR_DOCUMENT_EMBEDDING", default=15, cast=int)
# USE_PARALLEL_INFERENCE_QUEUE = config("USE_PARALLEL_INFERENCE_QUEUE", default=False, cast=bool)
# MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS = config("MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS", default=10, cast=int)
# USE_RAMDISK = config("USE_RAMDISK", default=False, cast=bool)
# RAMDISK_PATH = config("RAMDISK_PATH", default="/mnt/ramdisk", cast=str)
# RAMDISK_SIZE_IN_GB = config("RAMDISK_SIZE_IN_GB", default=1, cast=int)
MAX_RETRIES = config("MAX_RETRIES", default=3, cast=int)
# DB_WRITE_BATCH_SIZE = config("DB_WRITE_BATCH_SIZE", default=25, cast=int) 
# RETRY_DELAY_BASE_SECONDS = config("RETRY_DELAY_BASE_SECONDS", default=1, cast=int)
# JITTER_FACTOR = config("JITTER_FACTOR", default=0.1, cast=float)
# BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
embedding_model_cache = {} # Model cache to store loaded models



app = FastAPI(docs_url="/")  # Set the Swagger UI to root
# engine = create_async_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
# AsyncSessionLocal = sessionmaker(
#     bind=engine,
#     class_=AsyncSession,
#     expire_on_commit=False,
#     autoflush=False
# )
   
# # Core functions start here:  

async def get_or_compute_embedding(request: EmbeddingRequest, req: Request = None, client_ip: str = None) -> EmbeddingResponse:
    request_time = datetime.utcnow()  # Capture request time as datetime object

    model = load_model(DEFAULT_MODEL_NAME)
    
    embedding_list = calculate_sentence_embedding(model, request.input)
    
    if embedding_list is None:
        logger.error(f"Could not calculate the embedding for the given text: '{request.input}' !'")
        raise HTTPException(status_code=400, detail="Could not calculate the embedding for the given text")
    
    response_time = datetime.utcnow()  # Capture response time as datetime object
    total_time = (response_time - request_time).total_seconds() # Calculate total time using datetime objects
    
    word_length_of_input_text = len(request.input.split())
    if word_length_of_input_text > 0:
        logger.info(f"Embedding calculated for '{request.input}' in {total_time} seconds, or an average of {total_time/word_length_of_input_text :.2f} seconds per word")
    
    res = EmbeddingResponse(index=0, embedding= embedding_list)
    # return {"embedding": embedding_list}
    # return embedding_list
    return res

def load_model(model_name: str, raise_http_exception: bool = True):
    if model_name in embedding_model_cache:
        return embedding_model_cache[model_name]
    models_dir= DEFAULT_MODEL_DIR
    matching_files = glob.glob(os.path.join(models_dir, f"{model_name}*"))
    if not matching_files:
        logger.error(f"No model file found matching: {model_name}")
        raise FileNotFoundError
    matching_files.sort(key=os.path.getmtime, reverse=True)
    model_file_path = matching_files[0]
    logger.info(f"Loading the model from: {model_file_path}")
    try:
        if DEFAULT_MODEL_TYPE == 'LLAMA':
            model_instance = LlamaCppEmbeddings(model_path=model_file_path, use_mlock=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS)
            model_instance.client.verbose = False
        else:
            model_instance = BertCppEmbeddings(model_file_path)
            
        embedding_model_cache[model_name] = model_instance
        return model_instance
    
    except TypeError as e:
        logger.error(f"TypeError occurred while loading the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Exception occurred while loading the model: {e}")
        if raise_http_exception:
            raise HTTPException(status_code=404, detail="Model file not found")
        else:
            raise FileNotFoundError(f"No model file found matching: {model_name}")

# def load_token_level_embedding_model(model_name: str, raise_http_exception: bool = True):
#     try:
#         if model_name in token_level_embedding_model_cache: # Check if the model is already loaded in the cache
#             return token_level_embedding_model_cache[model_name]
#         models_dir = os.path.join(RAMDISK_PATH, 'models') if USE_RAMDISK else os.path.join(BASE_DIRECTORY, 'models') # Determine the model directory path
#         matching_files = glob.glob(os.path.join(models_dir, f"{model_name}*")) # Search for matching model files
#         if not matching_files:
#             logger.error(f"No model file found matching: {model_name}")
#             raise FileNotFoundError
#         matching_files.sort(key=os.path.getmtime, reverse=True) # Sort the files based on modification time (recently modified files first)
#         model_file_path = matching_files[0]
#         model_instance = Llama(model_path=model_file_path, embedding=True, n_ctx=LLM_CONTEXT_SIZE_IN_TOKENS, verbose=False) # Load the model
#         token_level_embedding_model_cache[model_name] = model_instance # Cache the loaded model
#         return model_instance
#     except TypeError as e:
#         logger.error(f"TypeError occurred while loading the model: {e}")
#         raise
#     except Exception as e:
#         logger.error(f"Exception occurred while loading the model: {e}")
#         if raise_http_exception:
#             raise HTTPException(status_code=404, detail="Model file not found")
#         else:
#             raise FileNotFoundError(f"No model file found matching: {model_name}")

# async def compute_token_level_embedding_bundle_combined_feature_vector(token_level_embeddings) -> List[float]:
#     start_time = datetime.utcnow()
#     logger.info("Extracting token-level embeddings from the bundle")
#     parsed_df = pd.read_json(token_level_embeddings) # Parse the json_content back to a DataFrame
#     token_level_embeddings = list(parsed_df['embedding'])
#     embeddings = np.array(token_level_embeddings) # Convert the list of embeddings to a NumPy array
#     logger.info(f"Computing column-wise means/mins/maxes/std_devs of the embeddings... (shape: {embeddings.shape})")
#     assert(len(embeddings) > 0)
#     means = np.mean(embeddings, axis=0)
#     mins = np.min(embeddings, axis=0)
#     maxes = np.max(embeddings, axis=0)
#     stds = np.std(embeddings, axis=0)
#     logger.info("Concatenating the computed statistics to form the combined feature vector")
#     combined_feature_vector = np.concatenate([means, mins, maxes, stds])
#     end_time = datetime.utcnow()
#     total_time = (end_time - start_time).total_seconds()
#     logger.info(f"Computed the token-level embedding bundle's combined feature vector computed in {total_time: .2f} seconds.")
#     return combined_feature_vector.tolist()

# async def get_or_compute_token_level_embedding_bundle_combined_feature_vector(token_level_embedding_bundle_id, token_level_embeddings, db_writer: DatabaseWriter) -> List[float]:
#     request_time = datetime.utcnow()
#     logger.info(f"Checking for existing combined feature vector for token-level embedding bundle ID: {token_level_embedding_bundle_id}")
#     async with AsyncSessionLocal() as session:
#         result = await session.execute(
#             select(TokenLevelEmbeddingBundleCombinedFeatureVector)
#             .filter(TokenLevelEmbeddingBundleCombinedFeatureVector.token_level_embedding_bundle_id == token_level_embedding_bundle_id)
#         )
#         existing_combined_feature_vector = result.scalar_one_or_none()
#         if existing_combined_feature_vector:
#             response_time = datetime.utcnow()
#             total_time = (response_time - request_time).total_seconds()
#             logger.info(f"Found existing combined feature vector for token-level embedding bundle ID: {token_level_embedding_bundle_id}. Returning cached result in {total_time:.2f} seconds.")
#             return json.loads(existing_combined_feature_vector.combined_feature_vector_json)  # Parse the JSON string into a list
#     logger.info(f"No cached combined feature_vector found for token-level embedding bundle ID: {token_level_embedding_bundle_id}. Computing now...")
#     combined_feature_vector = await compute_token_level_embedding_bundle_combined_feature_vector(token_level_embeddings)
#     combined_feature_vector_db_object = TokenLevelEmbeddingBundleCombinedFeatureVector(
#         token_level_embedding_bundle_id=token_level_embedding_bundle_id,
#         combined_feature_vector_json=json.dumps(combined_feature_vector)  # Convert the list to a JSON string
#     )
#     logger.info(f"Writing combined feature vector for database write for token-level embedding bundle ID: {token_level_embedding_bundle_id} to the database...")
#     await db_writer.enqueue_write([combined_feature_vector_db_object])
#     return combined_feature_vector
      
def calculate_sentence_embedding(model: Embeddings, text: str) -> np.array:
    sentence_embedding = None
    retry_count = 0
    while sentence_embedding is None and retry_count < 3:
        try:
            if retry_count > 0:
                logger.info(f"Attempting again calculate sentence embedding. Attempt number {retry_count + 1}")
            sentence_embedding = model.embed_query(text)
        except TypeError as e:
            logger.error(f"TypeError in calculate_sentence_embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Exception in calculate_sentence_embedding: {e}")
            text = text[:-int(len(text) * 0.1)]
            retry_count += 1
            logger.info(f"Trimming sentence due to too many tokens. New length: {len(text)}")
    if sentence_embedding is None:
        logger.error("Failed to calculate sentence embedding after multiple attempts")
    return sentence_embedding

# async def compute_embeddings_for_document(strings: list, model_name: str, client_ip: str) -> List[Tuple[str, np.array]]:
#     results = []
#     if USE_PARALLEL_INFERENCE_QUEUE:
#         logger.info(f"Using parallel inference queue to compute embeddings for {len(strings)} strings")
#         start_time = time.perf_counter()  # Record the start time
#         semaphore = asyncio.Semaphore(MAX_CONCURRENT_PARALLEL_INFERENCE_TASKS)
#         async def compute_embedding(text):  # Define a function to compute the embedding for a given text
#             try:
#                 async with semaphore:  # Acquire a semaphore slot
#                     request = EmbeddingRequest(text=text, model_name=model_name)
#                     embedding = await get_embedding_vector_for_string(request, client_ip=client_ip)
#                     return text, embedding["embedding"]
#             except Exception as e:
#                 logger.error(f"Error computing embedding for text '{text}': {e}")
#                 return text, None
#         results = await asyncio.gather(*[compute_embedding(s) for s in strings])  # Use asyncio.gather to run the tasks concurrently
#         end_time = time.perf_counter()  # Record the end time
#         duration = end_time - start_time
#         if len(strings) > 0:
#             logger.info(f"Parallel inference task for {len(strings)} strings completed in {duration:.2f} seconds; {duration / len(strings):.2f} seconds per string")
#     else:  # Compute embeddings sequentially
#         logger.info(f"Using sequential inference to compute embeddings for {len(strings)} strings")
#         start_time = time.perf_counter()  # Record the start time
#         for s in strings:
#             embedding_request = EmbeddingRequest(text=s, model_name=model_name)
#             embedding = await get_embedding_vector_for_string(embedding_request, client_ip=client_ip)
#             results.append((s, embedding["embedding"]))
#         end_time = time.perf_counter()  # Record the end time
#         duration = end_time - start_time
#         if len(strings) > 0:
#             logger.info(f"Sequential inference task for {len(strings)} strings completed in {duration:.2f} seconds; {duration / len(strings):.2f} seconds per string")
#     filtered_results = [(text, embedding) for text, embedding in results if embedding is not None] # Filter out results with None embeddings (applicable to parallel processing) and return
#     return filtered_results

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"message": "An unexpected error occurred"})

#FastAPI Endpoints start here:

@app.post("/get_embedding_vector_for_string/",
          response_model=EmbeddingResponse,
          summary="Retrieve Embedding Vector for a Given Text String",
          description="""Retrieve the embedding vector for a given input text string using the specified model.

### Parameters:
- `request`: A JSON object containing the input text string (`text`) and the model name.
- `token`: Security token (optional).

### Request JSON Format:
The request must contain the following attributes:
- `text`: The input text for which the embedding vector is to be retrieved.
- `model_name`: The model used to calculate the embedding (optional, will use the default model if not provided).

### Example (note that `model_name` is optional):
```json
{
  "text": "This is a sample text.",
  "model_name": "llama2_7b_chat_uncensored"
}
```

### Response:
The response will include the embedding vector for the input text string.

### Example Response:
```json
{
  "embedding": [0.1234, 0.5678, ...]
}
```""", response_description="A JSON object containing the embedding vector for the input text.")
async def get_embedding_vector_for_string(request: EmbeddingRequest, req: Request = None) -> EmbeddingResponse:
    try:
        return await get_or_compute_embedding(request, req)
    except Exception as e:
        logger.error(f"An error occurred while processing the request: {e}")
        logger.error(traceback.format_exc()) # Print the traceback
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.on_event("startup")
async def startup_event():
    # global db_writer, faiss_indexes, token_faiss_indexes, associated_texts_by_model
    # await initialize_db()
    queue = asyncio.Queue()
    # await db_writer.initialize_processing_hashes() 
    # list_of_downloaded_model_names = download_models()
    # for model_name in list_of_downloaded_model_names:
    try:
        load_model(DEFAULT_MODEL_NAME, raise_http_exception=False)
    except FileNotFoundError as e:
        logger.error(e)
    # faiss_indexes, token_faiss_indexes, associated_texts_by_model = await build_faiss_indexes()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=EMBEDDING_SERVER_LISTEN_PORT)
