from flask import Flask, request
import requests
import urllib.request
import json
import datetime
import datetime
import pytz

from main import getForecastByCity
from allCities import getForecast

app = Flask(__name__)

@app.route('/weather-by-city', methods=['GET'])
def get_weather_by_city():
    city = request.args.get('city')
    data = getForecastByCity(city)
    return json.dumps(data)

@app.route('/weather', methods=['GET'])
def get_weather():
    data = getForecast()
    return json.dumps(data)

if __name__ == '__main__':
    app.run(debug=True)
