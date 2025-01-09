# winepredictionapp

This repository contains the application for predicting wine quality using a pre-trained machine learning model. The application is implemented in Java and uses Docker for deployment.

## Features

- **Prediction**: Predicts wine quality based on input data from a `TestData.csv` file.
- **Cloud Deployment**: Runs on an AWS EC2 instance.
- **Dockerized Application**: Available as a Docker image for easy deployment.

## Prerequisites

- Java Development Kit (JDK)
- Maven
- Docker
- AWS Account
- Pre-trained model saved in an S3 bucket

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hannandesai/winepredictionapp.git
   cd winepredictionapp
   ```
2. Build the project:
   ```bash
   mvn clean package
   ```

## Usage

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t winepredictionapp .
   ```
2. Push the image to Docker Hub:
   ```bash
   docker push <your-dockerhub-username>/winepredictionapp
   ```
3. Pull and run the Docker image on an EC2 instance:
   ```bash
   docker pull <your-dockerhub-username>/winepredictionapp
   docker run -it winepredictionapp
   ```

### Without Docker

1. Run the application directly:
   ```bash
   mvn clean package
   mvn exec:java -Dexec.mainClass="org.example.winepredictionapp.App"
   ```

## Input

- The application reads input data from a `TestData.csv` file to predict wine quality.

## Output

- The application outputs the predicted wine quality for each input sample.
