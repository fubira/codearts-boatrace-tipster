# === Builder stage ===
FROM oven/bun:1 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    make g++ python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13

WORKDIR /app

# TS dependencies
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Python dependencies
COPY ml/pyproject.toml ml/uv.lock ml/
RUN cd ml && uv sync --frozen --python 3.13

# === Runtime stage ===
FROM oven/bun:1-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --link --from=ghcr.io/astral-sh/uv:0.10.12 /uv /usr/local/bin/

WORKDIR /app

# Python runtime (--link makes layer hash independent of base image changes)
COPY --link --chown=1000:1000 --from=builder /opt/python /opt/python

# Installed dependencies
COPY --link --chown=1000:1000 --from=builder /app/node_modules node_modules/
COPY --link --chown=1000:1000 --from=builder /app/ml/.venv ml/.venv/

# Source code
COPY --link --chown=1000:1000 package.json tsconfig.json ./
COPY --link --chown=1000:1000 src/ src/
COPY --link --chown=1000:1000 ml/pyproject.toml ml/uv.lock ml/
COPY --link --chown=1000:1000 ml/src/ ml/src/
COPY --link --chown=1000:1000 ml/scripts/ ml/scripts/

RUN mkdir -p data ml/models && chown -R bun:bun data ml/models

ENV TZ=Asia/Tokyo
ENV NODE_ENV=production

USER bun
CMD ["bun", "run", "src/cli/index.ts", "run"]
