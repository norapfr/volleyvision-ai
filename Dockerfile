FROM python:3.11-slim

WORKDIR /app

# dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# copia e instala dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copia el código fuente
COPY src/ ./src/
COPY data/annotations/ ./data/annotations/
COPY runs/ ./runs/
COPY .env .env

# puerto del dashboard
EXPOSE 8501

# comando de arranque
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]