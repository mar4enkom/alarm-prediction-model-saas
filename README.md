# SaaS Alarm Prediction Model
This is a Python project developed by our team to predict air alarms in all Ukrainian cities by analyzing weather and news reports from an ISW. The project consists of several components, including data collection, model training and improvement, and the development of a frontend and backend to support the model.

## Overview
The model is trained on historical data of the SaaS application's performance and is designed to predict when a future alarm may occur. The model is built using the scikit-learn library and is wrapped in a Flask REST API for easy integration into existing systems.

## Endpoints
```
1. http://43.206.107.170:8000/weather-by-city?city=yourCity
```
This endpoint predicts the alarm for the next 12 hours for a particular city. Replace yourCity in the URL with the name of the city you want to predict the alarm for.

```
2. http://43.206.107.170:8000/weather
```
This endpoint predicts the alarm for the next 12 hours for all regions in Ukraine.

## Getting Started

To get started with using the model, follow these steps:

1. Clone the repository:

```
git clone https://github.com/mar4enkom/saas-alarm-prediction-model.git
```

2. Install the required packages:

```
pip install -r requirements.txt
```

3. Start the Flask server:

```
python app.py
```

4. Use the API to make predictions:

```
curl -X POST -H "Content-Type: application/json" -d '{"cpu_utilization": 0.8, "memory_utilization": 0.7, "disk_utilization": 0.5}' http://localhost:5000/weather
```
or
```
curl -X POST -H "Content-Type: application/json" -d '{"cpu_utilization": 0.8, "memory_utilization": 0.7, "disk_utilization": 0.5}' http://localhost:5000/weather-by-city?city=yourCity
```
