#!/usr/bin/env python3
"""
This script retrieves and displays the first launch information using
the SpaceX API.
"""

import requests


def get_first_launch():
    """
    Fetches and displays information about the first
    SpaceX launch.

    Format:
        <launch name> (<date>) <rocket name> - <launchpad name>
        (<launchpad locality>)
    """
    url = "https://api.spacexdata.com/v4/launches"
    response = requests.get(url)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data: {response.status_code}")

    launches = response.json()

    # Sort launches by date_unix (ascending order)
    launches.sort(key=lambda x: x.get("date_unix", float('inf')))
    first_launch = launches[0]

    # Extract required information
    launch_name = first_launch.get("name", "Unknown Launch")
    launch_date = first_launch.get("date_local", "Unknown Date")
    rocket_id = first_launch.get("rocket")
    launchpad_id = first_launch.get("launchpad")

    # Fetch rocket details
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_response = requests.get(rocket_url)
    rocket_name = rocket_response.json().get("name", "Unknown Rocket")\
        if rocket_response.status_code == 200 else "Unknown Rocket"

    # Fetch launchpad details
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_response = requests.get(launchpad_url)
    if launchpad_response.status_code == 200:
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get("name", "Unknown Launchpad")
        launchpad_locality = launchpad_data.get("locality", "Unknown Locality")
    else:
        launchpad_name = "Unknown Launchpad"
        launchpad_locality = "Unknown Locality"

    # Format and print the result
    print(f"{launch_name} ({launch_date}) {rocket_name} - \
        {launchpad_name} ({launchpad_locality})")


if __name__ == "__main__":
    get_first_launch()
