FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl unzip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml download_data.sh ./
COPY src ./src
RUN pip install --no-cache-dir ".[mlflow]"

# Public datasets are downloaded and checksum-verified at build time (not stored in the repository).
RUN ./download_data.sh data/raw
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db MLFLOW_DISABLE_AGENT_HINT=1
RUN python -m pdm_ml.train --raw data/raw/cmapss --out models/latest \
    && python -m pdm_ml.export_demo --model models/latest --out /app/export/demo.json

ENV MODEL_DIR=/app/models/latest RAW_DIR=/app/data/raw/cmapss
EXPOSE 8000
CMD ["uvicorn", "pdm_ml.service:app", "--host", "0.0.0.0", "--port", "8000"]
