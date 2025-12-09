FROM python:3.12.7-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "-c", "print('Model churn ready!')"]