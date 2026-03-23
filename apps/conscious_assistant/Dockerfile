FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies from both root and app-specific requirements
COPY requirements.txt ./requirements.txt
COPY apps/conscious_assistant/requirements.txt ./apps_requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r apps_requirements.txt

# Copy full source (needed for relative imports like apps.conscious_assistant.*)
COPY . .

# Environment variables expected at runtime
ENV AGENT_MANIFEST_FILE="registries/manifest.hocon"
ENV AGENT_TOOL_PATH="coded_tools"

EXPOSE 5001

CMD ["python", "apps/conscious_assistant/interface_flask.py"]
