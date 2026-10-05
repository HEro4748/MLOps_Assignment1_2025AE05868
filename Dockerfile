FROM python:3.10-slim

WORKDIR /app

# Copy dependency definition and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, models, and data required for runtime
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Expose FastAPI default port
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]