FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
RUN pip install -U pip && pip install --no-cache-dir -e .
COPY src /app/src
COPY configs /app/configs
COPY data /app/data
EXPOSE 8000
CMD ["python", "-m", "ai_tweets.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
