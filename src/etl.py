import json
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
from datetime import datetime
import dotenv
import numpy as np
import requests
import bs4
from PIL import Image
from bs4 import BeautifulSoup
from chromadb.utils.data_loaders import ImageLoader
from tqdm import tqdm

from src.ai.tokenizers import tokenizer_factory
from src.db import get_collection

dotenv.load_dotenv()

BOILERPLATE_PATTERNS = [
    r'^Subscribe to',
    r'Click here to',
    r'Follow us on',
    r'Related posts?',
    r'Advertisement',
    r'©\s*\d{4}',
    r'Privacy Policy',
]
boiler_regex = [re.compile(pat, re.IGNORECASE) for pat in BOILERPLATE_PATTERNS]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

SITEMAP_URL = 'https://www.deeplearning.ai/sitemap.xml'
BATCH_PREFIX = 'https://www.deeplearning.ai/the-batch/'
# Listing pages, not articles: tag pages, newsletter issues, Andrew's letters
LISTING_PAGE = re.compile(r'/the-batch/(tag|issue-\d+|letters?)(/|$)')
LINKS_CACHE = Path('articles_links.txt')


def is_boilerplate(excerpt: str) -> bool:
    """
    Check if the given excerpt is boilerplate content.
    :param excerpt: Text excerpt to check.
    :return: True if the excerpt is considered boilerplate, False otherwise.
    """
    for rx in boiler_regex:
        if rx.search(excerpt):
            return True

    # skip if too short or not enough “real” content
    if len(excerpt.split()) < 3:
        return True

    return False


DATE = re.compile(r'[A-Z][a-z]{2} \d{1,2}, \d{4}')


def json_ld_article(soup: BeautifulSoup) -> dict:
    for script in soup.find_all('script', type='application/ld+json'):
        data = json.loads(script.string or '{}')
        if data.get('@type') in ('NewsArticle', 'Article'):
            return data
    return {}


def meta_content(soup: BeautifulSoup, prop: str) -> str:
    tag = soup.find('meta', property=prop) or soup.find('meta', attrs={'name': prop})
    return tag.get('content', '') if tag else ''


def format_date(iso_date: str) -> str:
    return datetime.fromisoformat(iso_date).strftime('%b %d, %Y') if iso_date else ''


def format_reading_time(duration: str) -> str:
    """ISO 8601 duration such as `PT5M` -> `5 min read`."""
    minutes = re.fullmatch(r'PT(\d+)M', duration or '')
    return f'{minutes[1]} min read' if minutes else ''


def article_tags(keywords: str) -> list[str]:
    # Older posts carry the publish date in `keywords` instead of real tags
    tags = (tag.strip() for tag in DATE.sub('', keywords).split(','))
    return [tag for tag in tags if tag]


def first_image(ld: dict, soup: BeautifulSoup) -> str:
    images = ld.get('image') or []
    if isinstance(images, str):
        images = [images]
    return images[0] if images else meta_content(soup, 'og:image')


def article_meta(soup: BeautifulSoup, canonical_url: str, related_links: list[str]) -> dict[str, str]:
    """
    Extract article metadata from JSON-LD, falling back to og:/article: meta tags.

    :return: Dictionary containing article metadata.
    """
    ld = json_ld_article(soup)
    return {
        'id': uuid.uuid4().hex,
        'title': ld.get('headline') or meta_content(soup, 'og:title'),
        'description': ld.get('description') or meta_content(soup, 'og:description'),
        'date': format_date(ld.get('datePublished') or meta_content(soup, 'article:published_time')),
        'tags': json.dumps(article_tags(ld.get('keywords', ''))),
        'reading_time': format_reading_time(ld.get('timeRequired', '')),
        'image': first_image(ld, soup),
        'related_links': json.dumps(related_links),
        'canonical_url': canonical_url,
    }


SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def split_to_fit(text: str, count_tokens, max_tokens: int) -> list[str]:
    """Split text longer than `max_tokens` at sentence boundaries."""
    if count_tokens(text) <= max_tokens:
        return [text]
    return [' '.join(group) for group in pack(SENTENCE_END.split(text), count_tokens, max_tokens)]


def pack(pieces: list[str], count_tokens, max_tokens: int) -> list[list[str]]:
    """Greedily group consecutive pieces so each group stays within `max_tokens`."""
    groups, current, current_tokens = [], [], 0
    for piece in pieces:
        tokens = count_tokens(piece)
        if current and current_tokens + tokens > max_tokens:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += tokens
    return groups + [current] if current else groups


def split_subtitle(text: str) -> tuple[str, str]:
    """Split a leading short "Subtitle: content" (e.g. "What's new: ...")."""
    head, sep, rest = text.partition(': ')
    return (head, rest) if sep and len(head.split()) < 20 else ('', text)


def make_token_chunks(
        meta: dict,
        excerpts: list[str],
        max_tokens: int = 510,
        tokenizer_provider: str = "huggingface",
) -> list[dict]:
    """
    Split the article content into token-limited chunks.

    :param meta: Metadata dict containing at least 'id' and 'date'.
    :param excerpts: List of text excerpts from the article.
    :param max_tokens: Maximum tokens per chunk, counted with the text embedder's tokenizer:
        its window is 512 including 2 special tokens, and longer input is silently truncated.
    :param tokenizer_provider: Which tokenizer to use (e.g. "openai", "huggingface").
    :return: List of dicts, each containing 'id', 'content', and 'meta' with subtitle, article_id, date, chunk_idx, total_chunks.
    """
    count_tokens = tokenizer_factory(tokenizer_provider).count_tokens
    pieces = [piece
              for excerpt in excerpts if not is_boilerplate(excerpt)
              for piece in split_to_fit(excerpt, count_tokens, max_tokens)]
    groups = pack(pieces, count_tokens, max_tokens)

    article_id = meta.get("id", "")
    chunks = []
    for idx, group in enumerate(groups):
        subtitle, content = split_subtitle(' '.join(group).strip())
        chunks.append({
            "id": f"{article_id}-{idx}",
            "content": content.strip(),
            "meta": {
                "article_id": article_id,
                "date": meta.get("date", ""),
                "subtitle": subtitle,
                "chunk_idx": idx,
                "total_chunks": len(groups),
            },
        })
    return chunks


def strip_widgets(body: bs4.Tag) -> None:
    for widget in body.select('iframe, #elevenlabs-audionative-widget, div[data-playerurl]'):
        widget.decompose()


def drop_trailing_promos(body: bs4.Tag) -> None:
    # Data Points posts put community promos after the first <hr>
    divider = body.find('hr', recursive=False)
    if divider is None:
        return
    for block in [divider, *divider.find_next_siblings()]:
        block.decompose()


def article_content(body: bs4.Tag) -> tuple[list[str], list[str]]:
    """
    Extract text excerpts (one per top-level block) and outbound links from the article body.

    :return: A tuple containing a list of text excerpts and a list of related links.
    """
    strip_widgets(body)
    drop_trailing_promos(body)
    related_links = [a['href'] for a in body.find_all('a', href=True)]
    blocks = (block.get_text(strip=True, separator=' ') for block in body.find_all(recursive=False))
    return [text for text in blocks if text], related_links


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.text, 'html.parser')


def scrape_article(url: str) -> tuple[dict, list[dict], dict]:
    """
    Scrape an article from the given URL and extract its metadata, content, and image.

    :return: A tuple containing:
        a dictionary with article metadata;
        a list of dictionaries representing text chunks with their metadata;
        a dictionary with image data (id and URL).
    """
    soup = fetch_soup(url)
    body = soup.select_one('article div.prose')
    if body is None:
        raise ValueError(f'No article body found at {url}')

    excerpts, related_links = article_content(body)
    meta = article_meta(soup, canonical_url=url, related_links=related_links)
    image = {
        'id': meta['id'],
        'url': meta.pop('image'),
        'meta': {'article_id': meta['id']},
    }
    return meta, make_token_chunks(meta, excerpts), image


def fetch_sitemap_links(url: str = SITEMAP_URL) -> list[str]:
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return re.findall(r'<loc>\s*([^<\s]+)\s*</loc>', response.text)


def is_article_link(link: str) -> bool:
    return (
        link.startswith(BATCH_PREFIX)
        and link != BATCH_PREFIX
        and not LISTING_PAGE.search(link)
    )


def collect_article_links() -> list[str]:
    links = {link.rstrip('/') + '/' for link in fetch_sitemap_links()}
    return sorted(filter(is_article_link, links))


def get_articles_links(limit: int | None = None) -> list[str]:
    """
    Collect The Batch article links from the public sitemap, cached in `articles_links.txt`.

    The Ghost Content API used previously now answers 402 "Site unavailable".

    :param limit: Optional cap on the number of links (handy for quick test runs).
    :return: List of article links.
    """
    if LINKS_CACHE.exists():
        links = LINKS_CACHE.read_text().splitlines()
    else:
        links = collect_article_links()
        LINKS_CACHE.write_text(''.join(f'{link}\n' for link in links))
    return links[:limit]


def image_url_to_pil(image_url: str, max_size: int | None = 512) -> Image.Image:
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
    images = get_collection(
        "images",
        data_loader=ImageLoader(),
    )
    articles = get_collection("articles")

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
                metadatas=[{**image['meta'], 'url': image['url']}],
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
