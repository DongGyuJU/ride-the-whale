FROM python:3.11-slim

# ?œìŠ¤???¨í‚¤ì§€ + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    libgomp1 \
    ca-certificates \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python ?˜ì¡´??ë¨¼ì? (ìºì‹œ ?¨ìœ¨)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ì½”ë“œ ë³µì‚¬
COPY . .

# ?˜ê²½ ë³€??(?°í??„ì— .env?ì„œ ??–´?°ê¸°)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# ?”í´??ëª…ë ¹
CMD ["python3", "-c", "print('Smart Money container ready. Use docker-compose run cli <command>')"]
