FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
                git \
                cmake \
                software-properties-common \
                python3.8 \
                python3-pip \
                libpq-dev \
                libspatialindex-dev

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt

WORKDIR /usr/src/app
COPY . .

#CMD [ "python", "./your-daemon-or-script.py" ]