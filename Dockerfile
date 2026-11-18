
FROM python:3.11-slim

# Python runtime configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Native libraries required by scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately for better build caching
COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Application source
COPY olin_app/ /app/olin_app/
COPY inference/ /app/inference/

# Required inference metadata and calibration assets
COPY datasets/dataset_v1/preprocessing_metadata.json \
    /app/datasets/dataset_v1/preprocessing_metadata.json

COPY training/results/v3_resnet18/target_normalization.json \
    /app/training/results/v3_resnet18/target_normalization.json

COPY training/results/v3_resnet18/uncertainty/calibration_factors.json \
    /app/training/results/v3_resnet18/uncertainty/calibration_factors.json

COPY FM1/dataset_generation_forward_model.py /app/FM1/dataset_generation_forward_model.py    

# Runtime directories (model and private files use persistent volumes)
RUN mkdir -p \
    /app/training/models \
    /app/private_storage/uploads \
    /app/private_storage/reports

EXPOSE 8000

CMD ["uvicorn", "olin_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
