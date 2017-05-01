# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:05:08 2017

@author: Leland
"""

class City:

    def __init__(self, nameP):
        self.name = nameP
        # List of destination cities
        self.routes = []

class Airport:

    def __init__(self, nameP, cityP):
        self.name = nameP
        self.city = cityP
        # List of destination airports
        self.routes = []       
# Main

airports = {}
cities = {}

# Parse Flu Trends data to produce dictionary of cities
fluFile = open('../smallData.csv',encoding='UTF-8').read()
lines = fluFile.split('\n')
firstLine = lines[0]
tokenized = firstLine.split(',')
for i in range(len(tokenized)-1):
    name = tokenized[i+1].split('-')[0]
    city = City(name)
    cities[name] = city


# Parse Airports File to produce dictionary of airports
airportFile = open('US_Airports.csv',encoding='UTF-8').read()
lines = airportFile.split('\n')
for row in lines:
    tokenized = row.split(',')
    if len(tokenized) == 14:
        name = tokenized[4]
        city = tokenized[2]
        if '\\' not in name:
            airport = Airport(name, city)
            airports[name] = airport


# Parse Routes File to add routes to airports
routeFile = open('Flight_Routes.csv',encoding='UTF-8').read()
lines = routeFile.split('\n')
for row in lines:
    tokenized = row.split(',')
    sourceName = tokenized[2]
    destinationName = tokenized[4]
    if sourceName in airports and destinationName in airports:  
        airport = airports.get(sourceName)
        destination = airports.get(destinationName)
        airport.routes.append(destination)
        destination.routes.append(airport)
        airports[sourceName] = airport
        airports[destinationName] = destination

    
# Add route information to city objects
for airportName in airports:
    airport = airports.get(airportName)
    routes = airport.routes
    cityName = airport.city
    if cityName in cities:
        city = cities.get(cityName)
        for destinationAirport in routes:
            destinationCity = cities.get(destinationAirport.city)
            if destinationAirport.city in cities:
                if not destinationCity in city.routes:
                    city.routes.append(destinationCity)
                if not city in destinationCity.routes:
                    destinationCity.routes.append(city)
                cities[cityName] = city
                cities[destinationAirport.city] = destinationCity

# Write graph structure to a csv file.  First column = source city. Following
# columns = edges to other cities
outputFile = open('CA_Graph_Input.csv', 'w')
for cityName in cities:
    city = cities.get(cityName)
    outputFile.write(cityName + ',')
    routes = city.routes
    for route in routes:
        outputFile.write(route.name + ','+str(len(cities.get(route.name).routes)) +',')
    outputFile.write('\n')
outputFile.close()