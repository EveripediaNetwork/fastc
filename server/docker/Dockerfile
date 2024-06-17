FROM python:3.12-slim-bookworm
WORKDIR /opt/app
COPY requirements.txt .
COPY server/requirements.txt ./requirements-server.txt
RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install libuv1 zlib1g \
    && apt-get clean && \
    pip install --upgrade pip && \
    pip install -r requirements.txt \
    pip install -r requirements-server.txt
COPY . .
RUN pip install .
ENTRYPOINT [ "./server/scripts/start-server.sh" ]
