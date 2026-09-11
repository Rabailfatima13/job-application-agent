# Plain Python + Streamlit image - no compiled dependencies in requirements.txt
# (pydantic, httpx, pypdf, anthropic, openai, mcp are all pure-Python wheels),
# so a slim base is enough; no build toolchain is installed.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Dependencies first, source second: a code change rebuilds only the fast
# COPY + nothing layer below, not a full pip install.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY job_agent/ job_agent/

# Real secrets and the SQLite data directory are never baked into the image -
# see docker-compose.yml's env_file and volume, and .dockerignore for .env.
EXPOSE 8501

CMD ["streamlit", "run", "job_agent/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
