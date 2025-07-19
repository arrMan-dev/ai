#FROM ubuntu:latest
#LABEL authors="arrismanduma"
#
#ENTRYPOINT ["top", "-b"]
# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy only the dependency files first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port your Flask app runs on (default 5000)
EXPOSE 5000

# Set environment variables that your app needs (e.g., to find the static folder)
# These will be overridden by GitLab CI/CD variables at runtime
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Command to run the Flask application using Gunicorn for production
# Gunicorn is a production-ready WSGI HTTP Server for Python.
# pip install gunicorn if not already in requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]