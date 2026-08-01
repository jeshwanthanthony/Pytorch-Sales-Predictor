FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    WORKSPACES_DIR=/var/lib/restaurant-forecast/workspaces

WORKDIR /app

COPY requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY api api
COPY collector collector
COPY dashboard dashboard
COPY database database
COPY pipelines pipelines
COPY training training

RUN mkdir -p /var/lib/restaurant-forecast/workspaces \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /var/lib/restaurant-forecast

USER app
EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
