FROM python:3.11-slim
WORKDIR /app

RUN apt update && apt install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pythie-serving-requirements.txt .
RUN pip install -r pythie-serving-requirements.txt

COPY . .
RUN pip install .[serving]

ENTRYPOINT [ "pythie-serving" ]
