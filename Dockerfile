# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.11.5
FROM python:${PYTHON_VERSION}-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Ensure Poetry is in PATH
ENV PATH="/root/.local/bin:$PATH"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Install project dependencies
RUN poetry install --no-dev

# Expose the port the app runs on
EXPOSE 3000

CMD ["poetry", "run", "python", "app.py"]