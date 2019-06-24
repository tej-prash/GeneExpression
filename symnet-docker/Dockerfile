FROM python:3.7.3-slim-stretch

RUN mkdir /symnet

WORKDIR /symnet

RUN apt-get update
RUN apt-get install -y --no-install-recommends gcc build-essential
RUN apt-get clean

RUN python3 -m pip install keras numpy sklearn matplotlib imageio pandas tensorflow
