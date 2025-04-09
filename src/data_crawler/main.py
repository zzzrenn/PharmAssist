import sys
import os
# Add the project root to path to resolve module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from typing import Any

from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities.typing import LambdaContext
from crawlers import NiceCrawler
from dispatcher import CrawlerDispatcher

logger = Logger(service="pharmassist/crawler")

_dispatcher = CrawlerDispatcher()
_dispatcher.register("nice", NiceCrawler)


def handler(event, context: LambdaContext | None = None) -> dict[str, Any]:
    for record in event.get("Records", []):
        link = record.get("body")
        crawler = _dispatcher.get_crawler(link)

        try:
            crawler.extract(link=link)
        except Exception as e:
            return {"statusCode": 500, "body": f"An error occurred: {str(e)}"}
    return {"statusCode": 200, "body": "Link processed successfully"}



if __name__ == "__main__":
    url = "https://www.nice.org.uk/guidance/ng106"

    event = {
        "Records": [
            {"body": url}
        ]
    }
    handler(event, None)

    # crawler = NiceCrawler()
    # crawler.extract(url)
