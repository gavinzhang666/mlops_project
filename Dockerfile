FROM python:3.10-slim

# Create app directory
WORKDIR /opt/program

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install flask gunicorn sagemaker-inference

# Copy all project files
COPY . .

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Expose the port SageMaker expects
EXPOSE 8080

# SageMaker will call "serve" to start the inference server
ENTRYPOINT ["python3", "serve"]
