# Start from a base Python 3.7 image
FROM python:3.7-slim

# Install Git
RUN apt-get update && \
    apt-get install -y git

# Install DVC
RUN pip install dvc[gs]

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app