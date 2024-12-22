# Use an official Python runtime
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy application files
COPY main.py .
COPY final_model.pkl .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
 
