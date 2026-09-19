FROM python:3.12-slim

WORKDIR /app

# Install system deps needed by some LLM Guard models
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the Needle 3 engine + weights into the image (cache: ~/.cache/
# cactus-needle) so runtime never fetches from Hugging Face. Non-fatal: an
# audit-only signal degrades to "unavailable" if the download fails.
RUN python - <<'EOF' || echo "WARNING: needle warmup failed; triage scanner will self-disable at runtime"
import needle
from typing import Literal
from pydantic import BaseModel

class Verdict(BaseModel):
    verdict: Literal["benign"]

needle.extract("warmup", Verdict)
print("needle warmup ok")
EOF

COPY app.py .

# Security posture: never phone home (NEEDLE_TELEMETRY=0); Needle engine +
# weights are baked at build. HF_HUB_OFFLINE is deliberately NOT set: the
# llm-guard DeBERTa model downloads at first use (pre-existing behavior).
ENV NEEDLE_TELEMETRY=0

ENV HOST=0.0.0.0
ENV PORT=8090

EXPOSE 8090

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
