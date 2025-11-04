# Base image from Python
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all files from current folder to container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask default port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
