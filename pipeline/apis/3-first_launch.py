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
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    r = requests.get(url)
    json = r.json()

    dates = [x['date_unix'] for x in json]
    index = dates.index(min(dates))
    next_launch = json[index]

    name = next_launch['name']
    date = next_launch['date_local']
    rocket_id = next_launch['rocket']
    launchpad_id = next_launch['launchpad']

    url_r = "https://api.spacexdata.com/v4/rockets/" + rocket_id
    req_r = requests.get(url_r)
    json_r = req_r.json()
    rocket_name = json_r['name']

    url_l = "https://api.spacexdata.com/v4/launchpads/" + launchpad_id
    req_l = requests.get(url_l)
    json_l = req_l.json()
    launchpad_name = json_l['name']
    launchpad_loc = json_l['locality']

    return (name + ' (' + date + ') ' + rocket_name + ' - ' +
            launchpad_name + ' (' + launchpad_loc + ')')


if __name__ == "__main__":
    print(get_first_launch())
