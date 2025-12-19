# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (required for Hugging Face Spaces)
EXPOSE 7860

# Command to run the FastAPI app
# We use filterproj.simp:app because your app object is in filterproj/simp.py
CMD ["uvicorn", "filterproj.simp:app", "--host", "0.0.0.0", "--port", "7860"]
