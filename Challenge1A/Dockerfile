FROM --platform=linux/amd64 python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY solution.py .

COPY trained_model.joblib .

RUN mkdir -p input output

CMD ["python", "solution.py", "/app/input", "/app/output"]
