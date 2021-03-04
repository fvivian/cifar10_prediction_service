FROM python:3.8-slim-buster

WORKDIR /home/src/
RUN mkdir /tmp/upload
RUN mkdir /home/resources

EXPOSE 5000

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /home/src
