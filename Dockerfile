# Stage 1: Install TS dependencies (changes when package.json/bun.lock change)
FROM oven/bun:1-debian AS install-ts

WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --production

# Stage 2: Install Python dependencies (changes when pyproject.toml/uv.lock change)
FROM ghcr.io/astral-sh/uv:debian AS install-py

WORKDIR /app/ml
COPY ml/pyproject.toml ml/uv.lock ./
RUN uv sync --frozen --no-dev

# Stage 3: Runtime
FROM oven/bun:1-debian

ENV TZ=Asia/Tokyo
WORKDIR /app

# System deps (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Dependencies (changes only when lock files change)
COPY --from=install-ts /app/node_modules ./node_modules
COPY package.json ./
COPY --from=install-py /app/ml/.venv ./ml/.venv
COPY ml/pyproject.toml ml/uv.lock ./ml/

# Runtime directories (rarely changes)
RUN mkdir -p data ml/models && chown -R bun:bun data ml/models

# Config files (rarely changes)
COPY tsconfig.json ./

# Python source (changes with ML code updates)
COPY ml/src ./ml/src
COPY ml/scripts ./ml/scripts

# TS source (changes most frequently)
COPY src ./src

USER bun
CMD ["bun", "run", "src/cli/index.ts", "run"]
