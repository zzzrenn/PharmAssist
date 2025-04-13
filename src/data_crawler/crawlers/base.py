from abc import ABC, abstractmethod
from tempfile import mkdtemp

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from core.db.documents import BaseDocument


class BaseAbstractCrawler(ABC):
    model: type[BaseDocument]

    def __init__(self, doc_limit: int = 3) -> None:
        self.doc_limit = doc_limit

        options = webdriver.ChromeOptions()

        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1280x1696")
        options.add_argument("--single-process")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-dev-tools")
        options.add_argument("--no-zygote")
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        options.add_argument("--remote-debugging-port=9222")
        options.binary_location = "/opt/chrome/chrome"

        self.set_extra_driver_options(options)

        self.driver = webdriver.Chrome(
            service=Service(executable_path="/opt/chromedriver"),
            options=options,
        )

    def set_extra_driver_options(self, options: Options) -> None:
        pass

    @abstractmethod
    def extract(self, link: str, **kwargs) -> None: ...
