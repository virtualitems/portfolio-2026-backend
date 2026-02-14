FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libpq-dev \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip3.11 install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY prompts/ ./prompts/
COPY staticfiles/ ./staticfiles/
COPY vision/ ./vision/

RUN mkdir -p mediafiles logs database .ultralytics

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 80

CMD ["python3.11", "-OO", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers", "--lifespan", "off"]

# docker build -t backend:latest .
# docker run -d --name backend -p 8000:80 --gpus all backend:latest