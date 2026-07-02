FROM python:3.12-slim

WORKDIR /app

# system deps for some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code (frontend/dist must be built and committed, or built here)
COPY . .

# HF Spaces expects the app on port 7860
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]