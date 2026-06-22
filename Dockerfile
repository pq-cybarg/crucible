# Multi-arch Crucible (backend + GUI). Lightweight control-plane image — runs on any
# Linux distro and on Raspberry Pi (arm64). torch/lm-eval are NOT bundled (the heavy
# abliteration adapter is opt-in: pip install torch transformers, or use a GPU node).
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim AS app
WORKDIR /app
COPY pyproject.toml ./
COPY backend ./backend
RUN pip install --no-cache-dir -e .
COPY --from=web /web/dist ./frontend/dist
ENV CRUCIBLE_DATA_DIR=/data CRUCIBLE_STATIC=/app/frontend/dist
VOLUME /data
EXPOSE 8400
CMD ["uvicorn", "crucible.app:app", "--host", "0.0.0.0", "--port", "8400"]
