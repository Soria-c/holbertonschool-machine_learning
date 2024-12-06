#!/usr/bin/env python3
"""
This script retrieves and displays the first SpaceX launch with details:
launch name, date, rocket name, and launchpad information.
"""

import requests
from datetime import datetime


def get_first_launch():
    """
    Fetches the first SpaceX launch from the API and returns
    the formatted details.

    Returns:
        str: The formatted string containing launch details,
        or an error message.
    """
    base_url = "https://api.spacexdata.com/v4"

    # Fetch all launches
    launches_response = requests.get(f"{base_url}/launches")
    if launches_response.status_code != 200:
        return "Error: Unable to fetch launches."
    launches = launches_response.json()

    # Find the earliest launch by date_unix
    earliest_launch = min(launches, key=lambda launch:
                          launch.get("date_unix", float("inf")))

    # Extract launch details
    launch_name = earliest_launch.get("name", "Unknown launch")
    date_unix = earliest_launch.get("date_unix")
    if date_unix is None:
        return "Error: Date missing for the earliest launch."
    launch_date = datetime.fromtimestamp(date_unix)\
        .strftime("%Y-%m-%d %H:%M:%S")

    rocket_id = earliest_launch.get("rocket")
    launchpad_id = earliest_launch.get("launchpad")

    # Fetch rocket details
    rocket_response = requests.get(f"{base_url}/rockets/{rocket_id}")
    rocket_name = rocket_response.json().get("name", "Unknown rocket") \
        if rocket_response.status_code == 200 else "Unknown rocket"

    # Fetch launchpad details
    launchpad_response = requests.get(f"{base_url}/launchpads/{launchpad_id}")
    if launchpad_response.status_code == 200:
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data.get("name", "Unknown launchpad")
        launchpad_locality = launchpad_data.get("locality", "Unknown locality")
    else:
        launchpad_name = "Unknown launchpad"
        launchpad_locality = "Unknown locality"

    # Format and return the output
    return f"{launch_name} ({launch_date}) {rocket_name} - \
        {launchpad_name} ({launchpad_locality})"


if __name__ == "__main__":
    print(get_first_launch())
