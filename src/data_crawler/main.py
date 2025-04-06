import sys
import os
# Add the project root to path to resolve module imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from crawlers import NiceCrawler


if __name__ == "__main__":
    crawler = NiceCrawler()
    url = "https://www.nice.org.uk/guidance/ng133"
    crawler.extract(url)
