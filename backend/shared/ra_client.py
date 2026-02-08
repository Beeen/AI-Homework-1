"""Resident Advisor GraphQL API client with BrightData proxy support."""

import os
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

RA_GRAPHQL_URL = "https://ra.co/graphql"

# ---------------------------------------------------------------------------
# GraphQL query strings (translated from ra-api.ts)
# ---------------------------------------------------------------------------

_ARTIST_QUERY = (
    "query GET_ARTIST_BY_SLUG($slug: String!) {"
    "  artist(slug: $slug) {"
    "    id name followerCount firstName lastName aliases"
    "    isFollowing coverImage contentUrl"
    "    facebook soundcloud instagram twitter bandcamp discogs website"
    "    urlSafeName pronouns"
    "    country { id name urlCode __typename }"
    "    residentCountry { id name urlCode __typename }"
    "    ...biographyFields __typename"
    "  }"
    "}"
    " fragment biographyFields on Artist {"
    "  id name contentUrl image"
    "  biography { id blurb content discography __typename }"
    "  __typename"
    "}"
)

_VENUE_QUERY = (
    "query GET_VENUE($id: ID!) {"
    "  venue(id: $id) {"
    "    id name logoUrl photo blurb address"
    "    isFollowing contentUrl phone website followerCount capacity"
    "    raSays isClosed"
    "    topArtists { name contentUrl __typename }"
    "    eventCountThisYear"
    "    area { id name urlName country { id name urlCode isoCode __typename } __typename }"
    "    location { latitude longitude __typename }"
    "    __typename"
    "  }"
    "}"
)

_PROMOTER_QUERY = (
    "query GET_PROMOTER_DETAIL($id: ID!) {"
    "  promoter(id: $id) {"
    "    id name contentUrl followerCount isFollowing"
    "    website email blurb logoUrl"
    "    socialMediaLinks { id link platform __typename }"
    "    area { id name urlName country { id name urlCode __typename } __typename }"
    "    tracking(types: [PAGEVIEW]) { id code event __typename }"
    "    __typename"
    "  }"
    "}"
)

# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def parse_ra_url(url: str) -> Dict[str, str]:
    """Parse an RA URL to extract entity type and identifier.

    Supported patterns:
      - https://ra.co/dj/<slug>        -> artist, slug
      - https://ra.co/clubs/<id>       -> venue, id
      - https://ra.co/promoters/<id>   -> promoter, id

    Returns:
        dict with ``entity_type`` and ``identifier`` keys.

    Raises:
        ValueError: if the URL is not a recognised RA pattern.
    """
    parsed = urlparse(url)
    if parsed.netloc not in ("ra.co", "www.ra.co"):
        raise ValueError(f"Not an RA URL: {url}")

    path = parsed.path.strip("/")

    m = re.match(r"^dj/([a-zA-Z0-9_-]+)$", path)
    if m:
        return {"entity_type": "artist", "identifier": m.group(1)}

    m = re.match(r"^clubs/(\d+)$", path)
    if m:
        return {"entity_type": "venue", "identifier": m.group(1)}

    m = re.match(r"^promoters/(\d+)$", path)
    if m:
        return {"entity_type": "promoter", "identifier": m.group(1)}

    raise ValueError(f"Unrecognised RA URL pattern: {url}")


# ---------------------------------------------------------------------------
# BrightData proxy helpers
# ---------------------------------------------------------------------------


def _get_brightdata_proxy() -> Optional[str]:
    """Build a BrightData proxy URL from environment variables."""
    host = os.getenv("BRIGHTDATA_PROXY_HOST")
    user = os.getenv("BRIGHTDATA_PROXY_USERNAME")
    pw = os.getenv("BRIGHTDATA_PROXY_PASSWORD")
    if host and user and pw:
        return f"http://{user}:{pw}@{host}"
    return None


# ---------------------------------------------------------------------------
# GraphQL fetch
# ---------------------------------------------------------------------------


def fetch_ra_entity(entity_type: str, identifier: str) -> Dict[str, Any]:
    """Fetch entity data from the RA GraphQL API via BrightData proxy.

    Args:
        entity_type: One of ``"artist"``, ``"venue"``, ``"promoter"``.
        identifier: The slug (artist) or numeric ID (venue/promoter).

    Returns:
        Raw GraphQL response data for the requested entity.

    Raises:
        ValueError: for unknown entity types.
        httpx.HTTPStatusError: on HTTP errors from the API.
        RuntimeError: when the response contains no usable data.
    """
    queries = {
        "artist": (_ARTIST_QUERY, "GET_ARTIST_BY_SLUG", {"slug": identifier}),
        "venue": (_VENUE_QUERY, "GET_VENUE", {"id": identifier}),
        "promoter": (_PROMOTER_QUERY, "GET_PROMOTER_DETAIL", {"id": identifier}),
    }

    if entity_type not in queries:
        raise ValueError(f"Unknown entity type: {entity_type}")

    query_str, op_name, variables = queries[entity_type]
    proxy_url = _get_brightdata_proxy()

    client_kwargs: Dict[str, Any] = {
        "timeout": 20.0,
        "headers": {
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://ra.co/",
        },
    }
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    with httpx.Client(**client_kwargs) as client:
        resp = client.post(
            RA_GRAPHQL_URL,
            json={
                "operationName": op_name,
                "query": query_str,
                "variables": variables,
            },
        )
        resp.raise_for_status()

    data = resp.json()

    if data.get("errors"):
        print(f"[RA_CLIENT] GraphQL errors: {data['errors']}")

    root_key = {"artist": "artist", "venue": "venue", "promoter": "promoter"}[entity_type]
    entity_data = (data.get("data") or {}).get(root_key)

    if not entity_data:
        raise RuntimeError(
            f"No data returned for {entity_type} '{identifier}'. "
            f"Response: {data}"
        )

    return entity_data


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------


def _resolve_ra_image(raw: Any) -> Optional[str]:
    """Extract a usable image URL from various RA image field formats."""
    if not raw:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return raw.get("filename") or raw.get("url")
    return None


def _extract_promoter_social(social_links: list, platform: str) -> Optional[str]:
    """Extract a social media URL from a promoter's socialMediaLinks array."""
    if not social_links:
        return None
    for link in social_links:
        if not isinstance(link, dict):
            continue
        if (link.get("platform") or "").lower() == platform.lower():
            return link.get("link")
    return None


def normalize_ra_data(entity_type: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a raw RA GraphQL response into a flat enrichment-friendly dict.

    All entity types produce the same top-level keys so that downstream code
    can treat them uniformly.
    """
    if entity_type == "artist":
        bio_obj = raw.get("biography") or {}
        return {
            "name": raw.get("name"),
            "page_type": "artist",
            "bio": bio_obj.get("blurb") or bio_obj.get("content"),
            "profile_picture": _resolve_ra_image(raw.get("image")),
            "cover_picture": raw.get("coverImage"),
            "instagram": raw.get("instagram"),
            "soundcloud": raw.get("soundcloud"),
            "facebook": raw.get("facebook"),
            "twitter": raw.get("twitter"),
            "website": raw.get("website"),
            "bandcamp": raw.get("bandcamp"),
            "discogs": raw.get("discogs"),
            "country": (raw.get("country") or {}).get("name"),
            "resident_country": (raw.get("residentCountry") or {}).get("name"),
            "follower_count": raw.get("followerCount"),
            "content_url": raw.get("contentUrl"),
            # Venue/promoter-only fields set to None for uniformity
            "address": None,
            "latitude": None,
            "longitude": None,
            "city": None,
            "capacity": None,
        }

    if entity_type == "venue":
        location = raw.get("location") or {}
        area = raw.get("area") or {}
        return {
            "name": raw.get("name"),
            "page_type": "venue",
            "bio": raw.get("blurb"),
            "profile_picture": raw.get("logoUrl"),
            "cover_picture": raw.get("photo"),
            "instagram": None,  # Venue query doesn't return social links
            "soundcloud": None,
            "facebook": None,
            "twitter": None,
            "website": raw.get("website"),
            "bandcamp": None,
            "discogs": None,
            "country": (area.get("country") or {}).get("name"),
            "resident_country": None,
            "follower_count": raw.get("followerCount"),
            "content_url": raw.get("contentUrl"),
            "address": raw.get("address"),
            "latitude": location.get("latitude"),
            "longitude": location.get("longitude"),
            "city": area.get("name"),
            "capacity": raw.get("capacity"),
        }

    if entity_type == "promoter":
        social_links = raw.get("socialMediaLinks") or []
        area = raw.get("area") or {}
        return {
            "name": raw.get("name"),
            "page_type": "promoter",
            "bio": raw.get("blurb"),
            "profile_picture": raw.get("logoUrl"),
            "cover_picture": None,
            "instagram": _extract_promoter_social(social_links, "instagram"),
            "soundcloud": _extract_promoter_social(social_links, "soundcloud"),
            "facebook": _extract_promoter_social(social_links, "facebook"),
            "twitter": _extract_promoter_social(social_links, "twitter"),
            "website": raw.get("website"),
            "bandcamp": None,
            "discogs": None,
            "country": (area.get("country") or {}).get("name"),
            "resident_country": None,
            "follower_count": raw.get("followerCount"),
            "content_url": raw.get("contentUrl"),
            "address": None,
            "latitude": None,
            "longitude": None,
            "city": area.get("name"),
            "capacity": None,
        }

    raise ValueError(f"Unknown entity type: {entity_type}")
