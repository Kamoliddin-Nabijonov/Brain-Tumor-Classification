FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY Trained_Models ./Trained_Models
EXPOSE 9696
ENV FLASK_APP=app.py
CMD ["gunicorn", "--bind", "0.0.0.0:9696", "app:app"]