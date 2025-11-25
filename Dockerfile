# Use a lightweight official Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY requirements.txt .
COPY .env .
COPY shl_individual_tests.csv .
COPY rag_api_fastapi.py .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI default port
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "rag_api_fastapi:app", "--host", "0.0.0.0", "--port", "8000","--workers","2"]