FROM python:3.7.3-slim-stretch

RUN mkdir /symnet

WORKDIR /symnet

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install keras numpy sklearn matplotlib imageio pandas tensorflow
