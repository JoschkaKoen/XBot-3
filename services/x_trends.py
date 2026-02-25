"""
Fetch current trending topics in Germany from getdaytrends.com.

No API key required — plain web scrape. Always returns a list (empty on any
failure) so callers never have to handle exceptions.
"""

import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

logger = logging.getLogger("german_bot.x_trends")


def get_germany_trends(max_trends: int = 20) -> List[Dict]:
    """
    Scrape current German trending topics.

    Returns a list of dicts:
        [{"name": "#Oktoberfest", "tweet_volume": "12.5K tweets"}, ...]

    Returns an empty list on any network or parsing error so the caller can
    fall back gracefully.
    """
    url = "https://getdaytrends.com/germany/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Could not reach getdaytrends.com: %s", exc)
        return []

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        trends: List[Dict] = []

        for a_tag in soup.find_all(
            "a", href=lambda x: x and "/germany/trend/" in x
        ):
            name = a_tag.get_text().strip()
            if not name or any(t["name"] == name for t in trends):
                continue

            # Try to grab tweet volume from the parent table cell
            td = a_tag.find_parent("td")
            volume = "unknown"
            if td:
                full_text = td.get_text(separator=" ", strip=True)
                if "tweets" in full_text.lower():
                    volume = full_text.split(name)[-1].strip()

            trends.append({"name": name, "tweet_volume": volume})

            if len(trends) >= max_trends:
                break

        if not trends:
            logger.warning("getdaytrends.com returned no trends (page structure may have changed).")

        return trends

    except Exception as exc:
        logger.warning("Failed to parse trends page: %s", exc)
        return []
