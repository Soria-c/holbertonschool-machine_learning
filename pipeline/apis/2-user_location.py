#!/usr/bin/env python3
"""
This script retrieves and prints the location of a specific GitHub user
using the GitHub API. It handles errors for user not found and rate limits.
"""

import requests
import sys
from datetime import datetime, timedelta


def get_user_location(api_url):
    """
    Retrieves the location of a GitHub user from the given API URL.

    Args:
        api_url (str): The full API URL for the GitHub user.

    Returns:
        str: The user's location or an appropriate error message.
    """
    response = requests.get(api_url)

    if response.status_code == 404:
        return "Not found"
    elif response.status_code == 403:  # Rate limit exceeded
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        reset_in = datetime.fromtimestamp(reset_time) - datetime.now()
        reset_minutes = max(0, int(reset_in.total_seconds() / 60))
        return f"Reset in {reset_minutes} min"
    elif response.status_code == 200:
        user_data = response.json()
        return user_data.get("location", "Location not specified")
    else:
        return f"Error: Unexpected status code {response.status_code}"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub_API_URL>")
        sys.exit(1)

    api_url = sys.argv[1]
    print(get_user_location(api_url))
