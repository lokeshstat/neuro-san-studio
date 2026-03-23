FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install root + UI requirements
COPY requirements.txt ./requirements.txt
COPY apps/conscious_assistant/requirements.txt ./ui_requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r ui_requirements.txt

# Copy full source (needed for package imports like apps.conscious_assistant.*)
COPY . .

# Connection defaults - override at runtime for Azure:
#   NEURO_SAN_SERVER_CONNECTION=https
#   NEURO_SAN_SERVER_HOST=<backend-fqdn-no-scheme>
#   NEURO_SAN_SERVER_HTTP_PORT=443
ENV NEURO_SAN_SERVER_CONNECTION=direct
ENV NEURO_SAN_SERVER_HOST=localhost
ENV NEURO_SAN_SERVER_HTTP_PORT=8080
ENV AGENT_MANIFEST_FILE=registries/manifest.hocon
ENV AGENT_TOOL_PATH=coded_tools

EXPOSE 5001

CMD ["python", "apps/conscious_assistant/interface_flask.py"]
