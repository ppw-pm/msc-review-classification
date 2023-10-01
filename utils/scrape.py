import argparse
import csv
import json
import logging

import requests
from bs4 import BeautifulSoup

"""
Scrape iniial data from trustpilot
"""


def scrapeTrustpilot(search_key, start_page, end_page):
    """
    Based on a search_key, start_page and end_page, scrape Netflix reviews from Trustpilot.

    The function will make an HTTP request and parse the HTML using BeautifulSoup. At the time of writing this, trustpilot exposes a <script id="__NEXT_DATA__"> tag that contains all of the reviews displayed on the website in JSON format. The tag was found when using the browser inspector.

    The JSON contains more information than that, and the results are paginated. That means for every page we need to send another request and parse the HTML. All reviews are saved after only saving some fields we are interested in for sentiment analysis.
    """
    # Scrape results from uk.trustpilot.com for www.netflix.com
    url = "https://uk.trustpilot.com/review/www.netflix.com"
    # The search string may change so add it dynamically
    search_str = f"&search={search_key}"

    # Count the amount of reviews scraped
    sum = 0
    results = []
    print(
        f"Starting scrape: scraping '{url}' and filtering for '{search_key}'",
    )

    for i in range(start_page, end_page + 1):
        resp = requests.get(f"{url}?page={i}{search_str}")
        soup = BeautifulSoup(resp.text, "html.parser")
        # Trust pilot has a <script> tag that fetches review information from an API.
        # Find the tag, get the raw text
        raw_data = soup.find_all("script", id="__NEXT_DATA__")[-1].get_text()

        # Parse raw_data as JSON
        data = json.loads(raw_data)
        reviews = data["props"]["pageProps"]["reviews"]

        for review in reviews:
            sum = sum + 1
            r = {
                "id": review["id"],
                "username": review["consumer"]["displayName"],
                "rating": review["rating"],
                "text": review["text"],
                "date": review["dates"]["publishedDate"],
                "location": review["consumer"]["countryCode"],
            }
            results.append(r)
    print(f"Finished scraping: scraped {sum} reviews")
    return results


def init_argparse() -> argparse.ArgumentParser:
    """Helps re-run the script multiple times with different parameters without changing the code over and over"""
    parser = argparse.ArgumentParser(
        usage="python %(prog)s [-h] [SEARCH_KEY] [--start] [--end]",
        description="""
    Script that will scrape reviews for 'www.netflix.com' from trustpilot's platform.The script should be run with a page range (start and end) and a keyword to filter reviews on. To run the scrape, the script makes use of an undocumented script tag in trustpilot's HTML page that lists review data in JSON format.

    Outputs a CSV file with all of the scrape reviews.
    """,
    )
    parser.add_argument(
        "search_key",
        help="Term used to filter the review search. It is also used in the name of the saved file",
    )

    parser.add_argument(
        "--start",
        type=int,
        help="Page in the list of reviews to start the scrape from (default: 1)",
        default=1,
    )
    parser.add_argument(
        "--end",
        type=int,
        help="Page in the list of reviews to stop the scrape at (default: 1)",
        default=1,
    )
    return parser


# Initialise script arguments
args_parser = init_argparse()
args = args_parser.parse_args()
start_page = args.start
end_page = args.end
search_key = args.search_key

results = scrapeTrustpilot(search_key, start_page, end_page)
# Use python CSV functions to save results into a file
with open(f"results/{search_key}_scrape_result.csv", mode="w") as result_file:
    # CSV requires column name to be declared first when using a dictionary to write out
    writer = csv.DictWriter(
        result_file, fieldnames=["id", "username", "rating", "text", "date", "location"]
    )
    # Write column names
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Saved scrape to results/{search_key}_scrape_results.csv")
