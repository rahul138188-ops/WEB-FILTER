# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for OpenCV 
# Note: libgl1-mesa-glx is replaced by libgl1 in newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# Fixed: "COPY . ." must be on one line with a space
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 7860 (required for Hugging Face Spaces)
EXPOSE 7860

# Command to run the FastAPI app
CMD ["uvicorn", "filterproj.simp:app", "--host", "0.0.0.0", "--port", "7860"]
