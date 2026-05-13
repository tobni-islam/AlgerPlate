FROM python:3.11-slim

WORKDIR /app

# 1. System libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip and install the Heavy dependencies first
# We use --default-timeout to prevent the network from cutting off too early
COPY requirements_heavy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements_heavy.txt

# 3. Install the API/Light dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the app
COPY models/ models/
COPY ocr/ ocr/
COPY api/ api/

EXPOSE 8000

# Use the full module path for uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]