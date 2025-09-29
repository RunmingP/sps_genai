# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy the dependency files first (better layer caching)
COPY pyproject.toml uv.lock /code/

# Install dependencies using uv (frozen -> respect uv.lock)
RUN uv sync --frozen

# --- If you added spaCy embeddings, make sure the model is available in the image ---
# Comment this out if you don't use spaCy embeddings
RUN uv run python -m spacy download en_core_web_md

# Copy the application code
COPY ./app /code/app

# Expose container port (FastAPI will run on 80 inside container)
EXPOSE 80

# Command to run the application
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]

