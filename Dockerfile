FROM python:3.12

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias primero
COPY requirements.txt /app/

# Actualizar pip y dependencias del sistema
RUN apt-get update && apt-get install -y gcc && pip install --upgrade pip

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Descargar modelo de spaCy
RUN python -m spacy download es_core_news_sm || true

# Copiar el resto del código
COPY . /app

# Exponer el puerto para Flask
EXPOSE 6000

# Comando para ejecutar la aplicación
CMD ["python", "modelo.py"]
