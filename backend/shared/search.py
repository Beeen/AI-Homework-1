"""Search API and text processing utilities."""

import os
import re
from typing import Optional
from urllib.parse import urlparse

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import llm

try:
    import validators
except ImportError:
    validators = None

SEARCH_TIMEOUT = 10.0


def _compact(text: str, limit: int = 200) -> str:
    """Compact text to a maximum length, truncating at word boundaries."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    truncated = cleaned[:limit]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(",.;- ")


def _search_api(query: str) -> Optional[str]:
    """Search the web using Tavily or SerpAPI if configured, return None otherwise."""
    query = query.strip()
    if not query:
        return None

    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_key,
                        "query": query,
                        "max_results": 3,
                        "search_depth": "basic",
                        "include_answer": True,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or ""
                snippets = [
                    item.get("content") or item.get("snippet") or ""
                    for item in data.get("results", [])
                ]
                combined = " ".join([answer] + snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass

    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key:
        try:
            with httpx.Client(timeout=SEARCH_TIMEOUT) as client:
                resp = client.get(
                    "https://serpapi.com/search",
                    params={
                        "api_key": serp_key,
                        "engine": "google",
                        "num": 5,
                        "q": query,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                organic = data.get("organic_results", [])
                snippets = [item.get("snippet", "") for item in organic]
                combined = " ".join(snippets).strip()
                if combined:
                    return _compact(combined)
        except Exception:
            pass

    return None


def _llm_fallback(instruction: str, context: Optional[str] = None) -> str:
    """Use the LLM to generate a response when search APIs aren't available."""
    prompt = "Respond with 200 characters or less.\n" + instruction.strip()
    if context:
        prompt += "\nContext:\n" + context.strip()
    response = llm.invoke(
        [
            SystemMessage(content="You are a concise travel assistant."),
            HumanMessage(content=prompt),
        ]
    )
    return _compact(response.content)


def _with_prefix(prefix: str, summary: str) -> str:
    """Add a prefix to a summary for clarity."""
    text = f"{prefix}: {summary}" if prefix else summary
    return _compact(text)


def _extract_url_from_text(text: str, platform: str) -> Optional[str]:
    """Extract a platform URL from text using regex patterns."""
    patterns = {
        "instagram": r"(?:https?://)?(?:www\.)?instagram\.com/([a-zA-Z0-9_\.]+)/?",
        "soundcloud": r"(?:https?://)?(?:www\.)?soundcloud\.com/([a-zA-Z0-9_-]+)/?",
        "spotify": r"(?:https?://)?open\.spotify\.com/artist/([a-zA-Z0-9]+)",
        "bandcamp": r"(?:https?://)?([a-zA-Z0-9-]+)\.bandcamp\.com/?",
        "website": r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})/?",
    }
    pattern = patterns.get(platform)
    if not pattern:
        return None
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        if platform == "instagram":
            return f"https://instagram.com/{match.group(1)}"
        elif platform == "soundcloud":
            return f"https://soundcloud.com/{match.group(1)}"
        elif platform == "spotify":
            return f"https://open.spotify.com/artist/{match.group(1)}"
        elif platform == "bandcamp":
            return f"https://{match.group(1)}.bandcamp.com"
        elif platform == "website":
            return f"https://{match.group(1)}"
    return None


def _validate_url(url: str) -> bool:
    """Check if a URL is valid and accessible."""
    if not url:
        return False
    if validators:
        return validators.url(url) is True
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False
