FROM python:3.10-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY src/models/ src/models/
COPY data/ data/
COPY src/models/main.py src/models/main.py

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install -r requirements.txt --no-cache-dir
#RUN pip install . --no-deps --no-cache-dir


CMD exec uvicorn src.models.main:app --port $PORT --host 0.0.0.0 --workers 1