
import requests
from bs4 import BeautifulSoup


def get_trends_Germany():

    # Specify the URL of the website you want to scrape
    url = 'https://trends24.in/germany/'

    # Send a GET request to the website
    response = requests.get(url)

    # Parse the response content as HTML
    soup = BeautifulSoup(response.content, 'html.parser')


    # Find all a tags and get their text
    tag_texts = [a.text for a in soup.find_all('a')]

    # Get only the first 20 entries
    first_20_texts = tag_texts[10:29]

    return first_20_texts