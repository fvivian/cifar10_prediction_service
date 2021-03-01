FROM python:3.8-slim-buster

WORKDIR /home/src/
RUN mkdir /tmp/upload
RUN mkdir /home/resources

COPY requirements.txt /home/requirements.txt
RUN pip3 install -r requirements.txt

COPY . /home/src
