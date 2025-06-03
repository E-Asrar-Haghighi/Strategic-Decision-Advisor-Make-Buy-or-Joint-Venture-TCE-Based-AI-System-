# Use a specific and well-maintained Python base image
FROM python:3.11.9-slim-bookworm

# Set environment variables for best practices
ENV PYTHONDONTWRITEBYTECODE 1  # Prevents python from writing .pyc files
ENV PYTHONUNBUFFERED 1         # Prevents python from buffering stdout/stderr

# Set work directory
WORKDIR /app

# Install system dependencies
# wkhtmltopdf is often used for PDF generation from HTML.
# If you are not generating PDFs this way, you might not need it.
# curl might be needed for some operations, but often not strictly for a basic Python app.
# Consider if these are truly necessary for your application's runtime.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wkhtmltopdf \
        # curl # Uncomment if explicitly needed by your app at runtime
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt

# Install Python dependencies
# Using a virtual environment inside Docker is generally not necessary
# as the container itself provides isolation.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# Ensure you have a .dockerignore file to exclude unnecessary files/folders (like venv, .git, __pycache__)
COPY . .
# Alternatively, be more specific if you have build artifacts or large unnecessary files in root:
# COPY ./agents ./agents
# COPY ./streamlit_app.py ./streamlit_app.py
# COPY ./main.py ./main.py
# COPY ./.env ./.env # If you want to bake .env into the image (less secure, use Docker secrets/env vars at runtime instead)
# COPY ./scenario_weights.json ./scenario_weights.json

# Expose the default Streamlit port
EXPOSE 8501

# Define the command to run your application
# Using --server.address=0.0.0.0 makes Streamlit accessible from outside the container.
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]