FROM python:3.11-slim-bookworm

# adding torchvision / Pillow 
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    libjpeg62-turbo-dev zlib1g-dev libpng-dev \
  && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /code

COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen

COPY ./app /code/app
COPY ./helper_lib /code/helper_lib

EXPOSE 80

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
