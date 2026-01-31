# MLflow Server with PostgreSQL and S3 support
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    mlflow==2.10.0 \
    psycopg2-binary \
    boto3 \
    redis \
    gunicorn

# Create non-root user
RUN useradd -m -u 1000 mlflow && \
    mkdir -p /mlflow/artifacts && \
    chown -R mlflow:mlflow /app /mlflow

USER mlflow

EXPOSE 5000

ENV MLFLOW_HOST=0.0.0.0
ENV MLFLOW_PORT=5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--app-name", "basic-auth"]
