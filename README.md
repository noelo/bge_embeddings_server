# FastAPI server providing embeddings using BGE models and BertCPP.

See https://huggingface.co/BAAI/bge-large-en


All of this is based on the great work of others:

1. BertCPP https://github.com/skeskinen/bert.cpp
2. llama_embeddings_fastapi_service https://github.com/Dicklesworthstone/llama_embeddings_fastapi_service
3. GGML models from https://huggingface.co/maikaarda/bge-base-en-ggml/tree/main



[!IMPORTANT] The BertCPP component is built using my cloned one as the GGML model format changed and the models weren't regenerated.
If the models and libggml don't match the libbertcpp core dumps

To run in a container

1. Download the GGML models from the link [3] above into a directory on your machine

2. Use _PodMan_  to pull and run the container
```
podman run -p 9090:8089  -v ./models/:/tmp/models/:ro,z -e DEFAULT_MODEL_DIR=/tmp/models https://quay.io/repository/noeloc/bge_embeddings_server
```

3. Use a browser or other tools to invoke a http post
```
curl -X 'POST' \
  'http://localhost:9090/get_embedding_vector_for_string/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "this a test ",
  "model": "ggml-model-q4_0"
}'
```