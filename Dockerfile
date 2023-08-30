# Use Python 3.9 image
FROM registry.access.redhat.com/ubi8/python-311

# Set environment variables
ENV PYTHONUNBUFFERED=1

USER 0

RUN mkdir /tmp/bert

WORKDIR /tmp/bert

RUN git clone https://github.com/noelo/bert.cpp

WORKDIR /tmp/bert/bert.cpp

RUN git submodule update --init --recursive

RUN mkdir build

WORKDIR /tmp/bert/bert.cpp/build

RUN cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release && make

RUN cp libbert.so /usr/lib64/

RUN cp ./ggml/src/libggml.so /usr/lib64/

RUN rm -rf /tmp/bert

USER 1001

# Set working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8089

# Command to run the application
CMD ["python3", "bge_embeddings_fastapi_server.py"]
