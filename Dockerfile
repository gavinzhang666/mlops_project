FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask gunicorn

COPY . .

RUN chmod +x /app/serve

EXPOSE 8080

ENTRYPOINT ["/app/serve"]
