import json
import os
import re
import time
import uuid
from io import BytesIO
from urllib.parse import unquote
import dotenv
import numpy as np
import requests
import bs4
from PIL import Image
from bs4 import BeautifulSoup
from chromadb.utils.data_loaders import ImageLoader
from tqdm import tqdm

from ai.embedders import NoOpEmbeddingFunction
from db import get_collection

dotenv.load_dotenv()


def article_meta(
        soup: bs4.PageElement | bs4.Tag | bs4.NavigableString,
        **kwargs
) -> dict[str, str]:
    """
    Extract metadata from the article soup.

    :param soup: BeautifulSoup object containing the article HTML.
    :keyword related_links: List of related links to include in the metadata.
    :keyword canonical_url: Canonical URL of the article.
    :return: Dictionary containing article metadata.
    """

    # h1 contains the title and a span with description
    h1 = soup.find('h1')
    description = h1.find('span')
    if description:
        description = description.extract().get_text(strip=True)
    else:
        description = ''

    # Remove the navigation bar to get cleaner links
    _nav = soup.find('nav').extract()

    links = soup.find_all('a')
    date_link = links.pop()  # The last link is the date

    # Get reading time
    reading_time = ' '.join(
        soup.find(string=re.compile(r'read')).parent.stripped_strings
    )

    # Account for lazy loading images (especially in Ghost CMS run on Next.js)
    noscript, image = soup.find('noscript'), None
    if noscript:
        image = noscript.find('img')
    image = image or soup.find('img')
    if image and image['src'].startswith('/_next/image/?url='):
        # If its Next.js lazy loaded image, extract the encoded URL
        # And remove parameters if present (because they clog the extension)
        image = (
            image['src'].replace('/_next/image/?url=', '')
            .split('&', 1)[0]
        )
    elif image:
        image = image['src']
    else:
        image = ''
    # Decode URL-encoded image path (if any)
    image = unquote(image)

    return {
        'id': uuid.uuid4().hex,
        'title': h1.get_text(strip=True),
        'description': description,
        'date': date_link.get_text(strip=True),
        'tags': json.dumps([link.get_text(strip=True) for link in links]),
        'reading_time': reading_time,
        'image': image,
        'related_links': json.dumps(kwargs.get('related_links', [])),
        'canonical_url': kwargs.get('canonical_url', ''),
    }


def make_chunks(
        meta: dict,
        excerpts: list[str],
        chunk_size: int or None = None
) -> list[dict]:
    """
    Split the article content into chunks of a specified size.

    :param meta: Metadata dictionary containing article information.
    :param excerpts: List of text excerpts from the article.
    :param chunk_size: Maximum size of each chunk in characters. If None, no chunking is applied.
    :return: List of dictionaries, each containing a chunk of text and its metadata.
    """
    chunks = []
    current_chunk = []
    article_id = meta.get('id', None)

    def process_current_chunk():
        if not current_chunk:
            return

        # Find out if the chunk is of the form "subtitle: content"
        parts = ' '.join(current_chunk).split(': ', 1)
        if len(parts) == 2:
            subtitle, content = parts[0], parts[1]
        else:
            subtitle, content = '', parts[0]

        chunks.append({
            'meta': {
                'subtitle': subtitle,
            },
            'id': f'{article_id}-{len(chunks)}',
            'content': content.strip()
        })
        current_chunk.clear()

    for excerpt in excerpts:
        # Skip empty excerpts or recommendations if they still exist
        if not excerpt.strip() or excerpt.startswith('Subscribe to'):
            continue

        next_iter_chunk_size = len(' '.join(current_chunk)) + len(excerpt)
        if not chunk_size or next_iter_chunk_size > chunk_size:
            process_current_chunk()
            current_chunk = [excerpt]
        else:
            current_chunk.append(excerpt)

    # Add the last chunk if it exists
    process_current_chunk()

    return chunks


def article_content(
        soup: bs4.PageElement | bs4.Tag | bs4.NavigableString
) -> tuple[list[str], list[str]]:
    """
    Extract text excerpts and related links from the article soup.
    This function extracts text from the article content, as well as outbound links.
    :param soup: BeautifulSoup object containing the article HTML.
    :return: A tuple containing a list of text excerpts and a list of related links.
    """

    # Clean ads and elevenlabs transcripts
    for iframe in soup.find_all('iframe'):
        iframe.decompose()

    elevenlabs_transcript = soup.find('div', attrs={'data-playerurl': True})
    if elevenlabs_transcript:
        elevenlabs_transcript.decompose()

    # Get related links
    related_links = list(map(
        lambda x: x['href'],
        # Filter out empty links and those without href attributes
        filter(lambda x: x and x.has_attr('href'), soup.find_all('a'))
    ))

    text_excerpts = list(
        # Filter out empty strings (they shouldn't be pipelined into the database)
        filter(
            lambda x: x,
            # Extract text from the soup and keep its structure
            map(lambda x: x.get_text(strip=True, separator=' '), soup.children)
        )
    )

    return text_excerpts, related_links


def scrape_article(url: str) -> tuple[dict, list[dict], dict]:
    """
    Scrape an article from the given URL and extract its metadata, content, and image.

    :param url: URL of the article to scrape.
    :return: A tuple containing:
        a dictionary with article metadata;
        a list of dictionaries representing text chunks with their metadata;
        a dictionary with image data (id and URL).;
    """

    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser').find('article')

    header = soup.find('header')
    content = soup.find('div', class_=re.compile(r'^post_postContent'))

    excerpts, related_links = article_content(content)
    meta = article_meta(
        header,
        related_links=related_links,
        canonical_url=url,
    )

    image = {
        # Image is one-to-one with article (in future may be with chunks too)
        'id': meta["id"],
        'url': meta.pop('image'),
    }

    return meta, make_chunks(meta, excerpts), image


def get_articles_links(api_limit: int = 100) -> list[str]:
    """
    Fetch links to articles from The Batch API and save them to a file.

    :param api_limit: Maximum number of articles to fetch per API call.
    :return: List of article links.
    """

    articles_file = 'articles_links.txt'
    junk_tags = ['letters', 'the-batch', 'hash-developer']
    # Base URL for The Batch
    base_url = 'https://www.deeplearning.ai/the-batch'
    # TODO: Consider switch to sitemap.xml parsing
    api_base_url = '{base}/?key={key}&limit={limit}&include={include}'.format(
        base='https://dl-staging-website.ghost.io/ghost/api/content/posts',
        key=os.getenv('THE_BATCH_API_KEY'),
        limit=api_limit,
        include='tags'  # authors,published
    )

    articles = set()

    # Try to read existing links from the file
    try:
        with open(articles_file, 'r') as f:
            return f.read().splitlines()
    except FileNotFoundError:
        print(f"File '{articles_file}' not found. Starting fresh.")

    pagination = {
        'next': '1',
    }
    print('Page size: {}'.format(api_limit))

    progress_bar = tqdm()
    while pagination['next'] is not None:
        response = requests.get(api_base_url + f'&page={pagination["next"]}')
        response.raise_for_status()
        response = response.json()
        posts, pagination = response['posts'], response['meta']['pagination']

        # Extract links from posts
        for post in posts:
            # Skip posts with a 'letters' or 'the-batch' tag
            if any(tag['slug'] in junk_tags for tag in post['tags']):
                continue

            if post['slug']:
                link = f"{base_url}/{post['slug']}"
                articles.add(link)

        # Show progress bar
        progress_bar.total = pagination['pages']
        progress_bar.update()

    # Save the links to a file
    with open(articles_file, 'w') as f:
        for link in articles:
            f.write(f"{link}\n")

    return list(articles)


def image_url_to_pil(image_url: str, max_size: int or None = 512) -> Image.Image:
    """
    Convert an image URL to a PIL image in bytes format.

    :param image_url: URL of the image to convert.
    :param max_size: Maximum size of the image (width or height) to resize to.
    :return: PIL Image object in RGB format.
    """
    response = requests.get(image_url)
    response.raise_for_status()

    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    if max_size:
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return pil_image


def run_pipeline():
    """
    Run the ETL pipeline to scrape articles, extract metadata, and store them in the database.
    :return: None
    """

    texts = get_collection("texts")
    images = get_collection("images", data_loader=ImageLoader())
    articles = get_collection("articles", embedding_function=NoOpEmbeddingFunction())

    def process_article(url: str) -> None:
        """
        Process a single article URL, scrape it, and store the data in the database.
        :param url: URL of the article to process.
        :return: None
        """

        meta, chunks, image = scrape_article(url)

        # Insert article metadata
        article_id, description = meta.pop("id"), meta.pop("description", "")
        articles.add(
            ids=[article_id],
            documents=[description],
            metadatas=[meta],
        )

        # Insert chunks
        texts.add(
            ids=[chunk['id'] for chunk in chunks],
            documents=[chunk['content'] for chunk in chunks],
            metadatas=[chunk['meta'] for chunk in chunks],
        )

        # Insert image if it exists
        if image['url']:
            images.add(
                ids=[image['id']],
                # Convert image URL to PIL image (required by OpenCLIP)
                # And convert it to numpy array  (required by ChromaDB)
                images=[np.array(image_url_to_pil(image['url']))],
                metadatas=[{'url': image['url']}],
            )

        # Sleep to be respectful & avoid hitting API limits
        time.sleep(2)

    links, failed_links = get_articles_links(), []
    for article in tqdm(
            links,
            unit="article",
            total=len(links),
            desc="Scraping articles"
    ):
        try:
            process_article(article)
        except (requests.HTTPError, requests.ConnectionError):
            failed_links.append(article)
        except Exception as e:
            print(f"Failed to process {article}: {e}")
            failed_links.append(article)

    if not failed_links:
        print("All articles processed successfully!")
        return

    print(f"Failed to process {len(failed_links)} articles:")
    print('\n'.join(failed_links))

    for failed_link in tqdm(
            failed_links,
            unit="article",
            desc="Retrying failed articles"
    ):
        try:
            process_article(failed_link)
        except (requests.HTTPError, requests.ConnectionError):
            print(f"Failed to process {failed_link} again. Skipping.")


if __name__ == "__main__":
    run_pipeline()
