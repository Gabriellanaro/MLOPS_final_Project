FROM python:3.10-slim

EXPOSE $PORT

WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY src/models/main.py src/models/main.py

RUN pip install fastapi
RUN pip install pydantic
RUN pip install uvicorn
RUN pip install sentencepiece
RUN pip install -r requirements.txt
RUN pip install sentencepiece

# CMD exec uvicorn src.models.main:app --port $PORT --host 127.0.0.1 --workers 1
CMD ["uvicorn","src.models.main:app","--port","8080","--host","0.0.0.0","--workers 1"]
