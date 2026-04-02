FROM oven/bun:1 AS base

# System deps: curl for scraping, python3 build tools, libgomp for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates make g++ python3 libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /usr/local/bin/

# Install Python via uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13

WORKDIR /app

# Install TS dependencies
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

# Install Python dependencies
COPY ml/pyproject.toml ml/uv.lock ml/
RUN cd ml && uv sync --frozen --python 3.13 && chown -R bun:bun .venv

# Copy source
COPY src/ src/
COPY ml/src/ ml/src/
COPY ml/scripts/ ml/scripts/
COPY tsconfig.json ./

ENV TZ=Asia/Tokyo
ENV NODE_ENV=production

RUN mkdir -p data ml/models && chown -R bun:bun .

USER bun
CMD ["bun", "run", "src/cli/index.ts", "run"]
