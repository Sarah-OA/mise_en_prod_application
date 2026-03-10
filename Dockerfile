FROM python:3.12-slim-bookworm

# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip

# Install project dependencies
COPY requirements.txt .
RUN uv add -r requirements.txt

COPY train.py .
COPY src ./src

CMD ["python3", "train.py"]
