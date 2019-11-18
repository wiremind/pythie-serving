FROM python:3.7-slim
WORKDIR /app

RUN apt update && apt install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install .

ENTRYPOINT [ "pythie-serving" ]
