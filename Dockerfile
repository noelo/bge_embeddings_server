# Use Python 3.9 image
FROM registry.access.redhat.com/ubi8/python-311

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies, including sudo
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libpq-dev \
#     libmagic1 \
#     sudo && \
#     rm -rf /var/lib/apt/lists/*

# # Upgrade pip and install wheel
# RUN python3 -m pip install --upgrade pip && \
#     python3 -m pip install wheel

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the .env file
COPY .env .

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8089

# Command to run the application
CMD ["python3", "llama_2_embeddings_fastapi_server.py"]
