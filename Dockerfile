# or lts-alpine or latest-alpine or slim
FROM python:3.9

# the commands will be executed within this directory (like cd command)
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "./main.py"]