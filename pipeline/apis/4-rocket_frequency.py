#!/usr/bin/env python3
"""
This script retrieves and displays the number of launches per
rocket using the SpaceX API.
"""

import requests
from collections import Counter


def launches_per_rocket():
    """
    Fetches the number of launches per rocket and prints
    them in the specified format.

    Format: <rocket name>: <launch count>
    """
    # Fetch all launches
    launches_url = "https://api.spacexdata.com/v4/launches"
    launches_response = requests.get(launches_url)
    if launches_response.status_code != 200:
        raise RuntimeError(f"Failed to fetch launches: \
            {launches_response.status_code}")
    launches = launches_response.json()

    # Count launches by rocket ID
    rocket_launch_counts = Counter(launch["rocket"] for launch in launches)

    # Fetch rocket details
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    rockets_response = requests.get(rockets_url)
    if rockets_response.status_code != 200:
        raise RuntimeError(f"Failed to fetch rockets: \
            {rockets_response.status_code}")
    rockets = rockets_response.json()

    # Map rocket IDs to names
    rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets}

    # Prepare and sort results
    results = [
        (rocket_names[rocket_id], count)
        for rocket_id, count in rocket_launch_counts.items()
        if rocket_id in rocket_names
    ]
    # Sort by count (desc) and name (asc)
    results.sort(key=lambda x: (-x[1], x[0]))

    # Print results
    for name, count in results:
        print(f"{name}: {count}")


if __name__ == "__main__":
    launches_per_rocket()
