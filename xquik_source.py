import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://xquik.com/api/v1"
DEFAULT_LIMIT = 5
MAX_LIMIT = 25
TEXT_FIELDS = ("text", "fullText", "full_text", "content")


class XquikSourceError(RuntimeError):
    """Raised when live tweet loading cannot complete."""


def _candidate_items(payload):
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return []

    for key in ("tweets", "items", "results", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _candidate_items(value)
            if nested:
                return nested

    return []


def _candidate_text(item):
    if not isinstance(item, dict):
        return None

    for field in TEXT_FIELDS:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    legacy = item.get("legacy")
    if isinstance(legacy, dict):
        value = legacy.get("full_text")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def parse_tweet_texts(payload, limit=DEFAULT_LIMIT):
    safe_limit = max(1, min(int(limit), MAX_LIMIT))
    texts = []

    for item in _candidate_items(payload):
        text = _candidate_text(item)
        if text is None:
            continue

        texts.append(text)
        if len(texts) >= safe_limit:
            break

    return texts


def fetch_xquik_tweets(
    query,
    *,
    api_key=None,
    limit=DEFAULT_LIMIT,
    base_url=BASE_URL,
    opener=urlopen,
):
    clean_query = query.strip()
    if not clean_query:
        raise XquikSourceError("Enter a search query before loading tweets.")

    token = api_key or os.getenv("XQUIK_API_KEY")
    if not token:
        raise XquikSourceError("Set XQUIK_API_KEY before loading live tweets.")

    url = f"{base_url.rstrip('/')}/x/tweets/search?{urlencode({'q': clean_query})}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "tweet-sentiment-simple",
            "x-api-key": token,
        },
    )

    try:
        with opener(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise XquikSourceError(f"Xquik request failed with HTTP {exc.code}.") from exc
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise XquikSourceError("Xquik request failed. Try again later.") from exc

    return parse_tweet_texts(payload, limit=limit)
