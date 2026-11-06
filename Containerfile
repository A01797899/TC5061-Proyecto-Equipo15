
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN python -m pip install --no-cache-dir \
    -r requirements-api.txt

RUN useradd --create-home \
    --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
