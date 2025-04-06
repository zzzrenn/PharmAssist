import time
from abc import ABC, abstractmethod
from tempfile import mkdtemp

from core.db.documents import BaseDocument
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class BaseAbstractCrawler(ABC):
    model: type[BaseDocument]

    def __init__(self, doc_limit: int = 3) -> None:
        self.doc_limit = doc_limit

        options = webdriver.ChromeOptions()

        options.add_argument("--no-sandbox")
        options.add_argument("--headless=new")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--log-level=3")
        options.add_argument("--disable-popup-blocking")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-networking")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        options.add_argument("--remote-debugging-port=9226")

        self.set_extra_driver_options(options)

        self.driver = webdriver.Chrome(
            options=options,
        )

    def set_extra_driver_options(self, options: Options) -> None:
        pass

    @abstractmethod
    def extract(self, link: str, **kwargs) -> None: ...