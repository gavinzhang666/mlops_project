FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt flask

COPY . .

EXPOSE 8080

ENTRYPOINT ["./serve"]
