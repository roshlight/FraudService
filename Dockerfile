FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY artifacts /app/artifacts
COPY src /app/src

RUN mkdir -p /app/input /app/output

CMD ["python", "-m", "src.main"]
