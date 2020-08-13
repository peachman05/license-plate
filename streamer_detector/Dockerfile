FROM python:3
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . /streamer_detector/
WORKDIR /streamer_detector/