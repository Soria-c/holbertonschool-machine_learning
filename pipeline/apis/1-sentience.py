#!/usr/bin/env python3
"""
This script uses the SWAPI API to fetch and filter starships based
on passenger capacity.
"""

import requests


def availableShips(passengerCount):
    """
    Fetches a list of starships that can hold at least
    a given number of passengers.

    Args:
        passengerCount (int): The minimum number of passengers the ship
        must hold.


    Returns:
        list: A list of starship names that meet the passenger requirement.
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch data from SWAPI: \
                {response.status_code}")

        data = response.json()
        for ship in data.get("results", []):
            # Extract passenger capacity, ignoring "unknown" values
            passengers = ship.get("passengers", "unknown").replace(",", "")
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship["name"])

        # Move to the next page if available
        url = data.get("next")

    return ships
