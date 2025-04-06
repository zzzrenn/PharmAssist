
import re

from aws_lambda_powertools import Logger
from crawlers.base import BaseAbstractCrawler

logger = Logger(service="pharmassist/crawler")


class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawlers = {}

    def register(self, domain: str, crawler: type[BaseAbstractCrawler]) -> None:
        self._crawlers[r"https://(www\.)?{}.org.uk/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseAbstractCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            logger.error(
                f"No crawler found for {url}. Defaulting to CustomArticleCrawler."
            )

            raise Exception(f"No crawler found for {url}")
