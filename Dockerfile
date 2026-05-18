# ---- Build stage ----
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install --prefix=/install packaging

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---- Final stage ----
FROM python:3.11-slim-bookworm

WORKDIR /app

# Only copy runtime libs, not build tools
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY *.py ./
COPY start.sh .

RUN chmod +x start.sh

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["bash", "start.sh"]