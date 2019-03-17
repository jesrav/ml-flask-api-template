# ML Flask API template
A template for exposing a ml-model in an API endpoint.

The ml-model being exposed is a simple k-nearest neighbors model,
predicting the species of Iris flower, trained on the classic Iris dataset.     

## Requirements
* Python 3.6 
* [Docker](https://www.docker.com/)

## Install

``` Bash
pipenv install && pipenv install --dev
```

Copy the example .env file.env file

``` Bash
cp .env.example .env
```

## Run the API locally by running
``` Bash
pipenv run start
```
## .. or building a docker image and runing a docker container exposing the api.
``` Bash
docker build . -t ml-flask-api-tmplate:latest
docker run -p 5000:5000 ml-flask-api-tmplate:latest
``` 
To interact with the api through the swagger ui, go to [http://localhost:5000/apidocs](http://localhost:5000/apidocs).
## Run script that trains the model on the Iris dataset
``` Bash
pipenv run train
```


