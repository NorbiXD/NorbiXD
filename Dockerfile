FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[postgres]"
COPY challenge.yaml ./
RUN useradd --create-home darwin && chown -R darwin /app
USER darwin
EXPOSE 8000
ENTRYPOINT ["darwin"]
CMD ["run", "--mode", "sim"]
