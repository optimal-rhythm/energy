# Energy Efficiency ML Prediction Service

## Problem Statement

The goal of this challenge is to build a regression model and deploy it with docker. The dataset you will use for the challenge is available at https://archive.ics.uci.edu/ml/datasets/Energy+efficiency. You should be able to run the docker image and then curl the container by sending json containing the attributes of a new building and get a json response with the heating and cooling loads predicted by your trained model. The code should be written in python but you can use whichever libraries you like to train and deploy the model.

## Installation

This project use two Docker images using an Anaconda base image - one for training the model and one for the prediction service.

The images can be built using

```sh
$ docker image build -t energy-train:1.0 .
```

and


```sh
$ docker image build -t energy-predict:1.0 .
```

from the respective folders.

## Training the model

The model training can be executed simply by running the training container.

```sh
$ docker container run energy-train:1.0
```

## Running the prediction service

The service is built using Flask and can be started by running the prediction container.

```sh
$ docker container run --publish 8000:8080 energy-predict:1.0
```

## Testing the prediction service

The service can be tested using a standard `curl` command or a tool such as Postman.

```sh
$ curl -d '{"RC" : 0.690, "SA" : 745.000, "WA" : 294.000, "RA": 220.5, "OH": 13.50, "OT": "north", "GA": 0.25, "GAD": "east"}' -H "Content-Type: application/json" -X POST http://localhost:8000/load/

{"HL": 16.69733333333333, "CL": 24.360999999999997}
```