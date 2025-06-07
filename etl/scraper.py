import time
from typing import Optional

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import bs4
from bs4 import BeautifulSoup
from selenium.webdriver.support.wait import WebDriverWait
from tqdm import tqdm

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:139.0) Gecko/20100101 Firefox/139.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br, zstd',
}

# Base URL for The Batch
BASE_URL = 'https://www.deeplearning.ai/the-batch'

tag_prefix = '/the-batch/tag/'
tags = [
    'tech-society',
    'research',
    'business',
    'data-points',
    'science',
    'hardware',
    # 'interviews-essays'
    'culture',
]

# Articles list file
articles_list = 'articles_links.txt'


def article_meta(article: bs4.PageElement | bs4.Tag | bs4.NavigableString):
    pass


def article_content(url: str):
    pass


def scrape_article(url: str):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')


def scrape_tag(tag: str, driver: Optional[WebDriver]=None):
    tag_url = f"{BASE_URL}/tag/{tag}"

    # If no driver is provided,
    # we will create a new one and dispose it after use
    disposable = driver is None

    driver = driver or webdriver.Chrome()
    wait = WebDriverWait(driver, 5)

    # Navigate to the tag URL
    driver.get(tag_url)

    progress_bar = tqdm(
        total=0, unit=" clicks", dynamic_ncols=True, smoothing=0.3
    )
    while True:
        try:
            # Wait for the 'button' to be clickable
            load_more_btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(text(), 'Load More')]"))
            )

            # Try clicking
            driver.execute_script("arguments[0].click();", load_more_btn)
            progress_bar.update()
        except (NoSuchElementException, TimeoutException):
            # If the button is not found, break the loop
            break

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Close the browser if we created it
    if disposable:
        driver.quit()

    articles = soup.find_all('article')
    links = [
        # First link is the tag, second link is the article URL
        article.find('a').find_next_sibling('a').get('href')
        for article in articles
    ]

    return links


def get_articles_links():
    articles = set()

    # Try to read existing links from the file
    try:
        with open(articles_list, 'r') as f:
            existing_links = f.read().splitlines()
            articles.update(existing_links)
            return articles
    except FileNotFoundError:
        print(f"File '{articles_list}' not found. Starting fresh.")

    # Set up Chrome options
    options = Options()
    options.add_argument("--headless=new")  # modern headless mode (since Chrome 109+)
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")  # Optional but recommended
    options.add_argument("--no-sandbox")  # Often needed for CI environments
    options.add_argument("--disable-dev-shm-usage")  # Fix for some memory issues in containers

    # Launch a browser
    driver = webdriver.Chrome(options=options)

    for tag in tags:
        print(f"Scraping tag '{tag}'...")
        links = scrape_tag(tag, driver=driver)
        print(f"Found {len(links)} articles in tag '{tag}'")
        articles.update(links)

    # Close the browser
    driver.quit()

    # Save the links to a file
    with open(articles_list, 'w') as f:
        for link in articles:
            f.write(f"{link}\n")

    return articles


def scrape_next():
    """
    Scrape the next batch of articles.
    This function is a placeholder for future scraping logic.
    """
    links = get_articles_links()

    # Placeholder for future scraping logic
    print("Scraping next batch of articles... (not implemented yet)")


scrape_next()