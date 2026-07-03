FROM python:3.11-slim

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Backend package manifest files for caching
COPY backend/package*.json ./backend/

# Install Backend dependencies
RUN cd backend && npm install

# Copy the rest of the project source files
COPY . .

# Expose port (Render automatically sets PORT environment variable, defaults to 5000)
EXPOSE 5000

# Fix potential Windows CRLF line ending issues in start.sh
RUN sed -i -e 's/\r$//' start.sh

# Make start.sh executable
RUN chmod +x start.sh

# Run start script
CMD ["./start.sh"]
