#!/usr/bin/env python3
"""
This script uses the SWAPI API to fetch the names of home planets
of all sentient species.
"""

import requests


def sentientPlanets():
    """
    Fetches the list of home planet names for all sentient species.

    Returns:
        list: A list of names of home planets of sentient species.
    """
    url = "https://swapi.dev/api/species/"
    sentient_planets = set()  # Use a set to avoid duplicates

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch data from SWAPI: \
                {response.status_code}")

        data = response.json()
        for species in data.get("results", []):
            # Check for "sentient" in classification or designation
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()
            if "sentient" in classification or "sentient" in designation:
                # Add home planet name if available
                homeworld_url = species.get("homeworld")
                if homeworld_url:
                    planet_response = requests.get(homeworld_url)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        sentient_planets.add(
                            planet_data.get("name", "Unknown"))

        # Move to the next page if available
        url = data.get("next")

    return list(sentient_planets)
