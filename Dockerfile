FROM tensorflow/tensorflow:2.5.0

RUN apt update && \
    apt-get install git -y && \
    apt install zstd

WORKDIR /app

COPY . /app

RUN mkdir ./checkpoints && \
    git clone https://github.com/kingoflolz/mesh-transformer-jax && \
    pip install -r requirements.txt && \
    pip install mesh-transformer-jax/ jax==0.2.12

RUN curl -LO "http://batbot.tv/ai/models/GPT-J-6B/step_383500_slim.tar.zstd"

RUN tar -I zstd -xf step_383500_slim.tar.zstd

RUN mv ./step_383500/ ./checkpoints/step_383500/

ENTRYPOINT ["python3"]

CMD ["main.py ""My name is"" "]
