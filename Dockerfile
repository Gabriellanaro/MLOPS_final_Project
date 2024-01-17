FROM python:3.10-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install -r requirements.txt

COPY src/models/main.py src/models/main.py

CMD exec uvicorn main:app --port $PORT --host 127.0.0.1 --workers 1