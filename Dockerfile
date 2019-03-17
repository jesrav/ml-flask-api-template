FROM ubuntu:18.04

RUN apt update && apt upgrade -y && apt install -y \
             python3-pip sqlite3

ENV LANG C.UTF-8

RUN pip3 install --upgrade pip
RUN pip3 install pipenv

# Create app folder
RUN mkdir -p /var/app
WORKDIR /var/app
COPY . /var/app

# Install requirements
RUN pipenv install --system

# Enviroment variables
ENV FLASK_APP=api.py

EXPOSE 5000

ENTRYPOINT [ "gunicorn", "--threads", "4", "-b", ":5000", "--access-logfile", "-", "--error-logfile", "-" , "api:app" ]