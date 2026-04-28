# Use the official lightweight Python image.
# Python 3.11 is a great stable choice.
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED True

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Run the web service on container startup using Uvicorn.
# Cloud Run automatically sets the PORT environment variable (default is 8080).
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}