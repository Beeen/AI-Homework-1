"""RA Entity Enrichment Pipeline — multi-agent graph for enriching Resident Advisor entities.

Accepts an RA URL, fetches structured data via the RA GraphQL API, identifies
missing fields, dispatches parallel discovery agents (Instagram via HikerAPI,
SoundCloud via SoundCloud API), and assembles a complete enriched profile.
"""

import json
import operator
import re
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from shared import (
    llm,
    _search_api,
    _compact,
    _extract_url_from_text,
    _validate_url,
    _TRACING,
    trace,
    using_prompt_template,
    using_attributes,
    parse_ra_url,
    fetch_ra_entity,
    normalize_ra_data,
    search_instagram_accounts,
    get_instagram_profile,
    search_soundcloud_users,
)


# =============================================================================
# STATE SCHEMA
# =============================================================================


class EnrichmentState(TypedDict):
    """State for the RA enrichment agent graph."""

    messages: Annotated[List[BaseMessage], operator.add]
    enrich_request: Dict[str, Any]

    # URL parsing
    ra_entity_type: Optional[str]
    ra_identifier: Optional[str]

    # RA GraphQL data
    ra_data: Optional[Dict[str, Any]]
    entity_name: Optional[str]

    # Gap analysis
    missing_fields: Optional[List[str]]

    # Discovery results
    instagram_result: Optional[Dict[str, Any]]
    soundcloud_result: Optional[Dict[str, Any]]

    # Resolution results
    profile_picture: Optional[Dict[str, Any]]
    bio: Optional[Dict[str, Any]]

    # Final output
    final_profile: Optional[Dict[str, Any]]

    # Observability
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]


# =============================================================================
# HELPERS
# =============================================================================


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM output, stripping markdown code fences if present."""
    if not text:
        return None
    # Strip ```json ... ``` wrappers
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


# =============================================================================
# NODE 1: RA SCRAPER
# =============================================================================


def ra_scraper_node(state: EnrichmentState) -> dict:
    """Parse the RA URL and fetch entity data via the GraphQL API."""
    req = state["enrich_request"]
    url = req["url"]
    calls: List[Dict[str, Any]] = []

    # --- Parse URL ---
    try:
        parsed = parse_ra_url(url)
        entity_type = parsed["entity_type"]
        identifier = parsed["identifier"]
        calls.append({
            "agent": "ra_scraper", "tool": "parse_ra_url",
            "args": {"url": url}, "result": parsed,
        })
    except ValueError as e:
        calls.append({"agent": "ra_scraper", "tool": "parse_ra_url", "error": str(e)})
        return {
            "messages": [SystemMessage(content=f"URL parse error: {e}")],
            "ra_entity_type": None,
            "ra_identifier": None,
            "ra_data": None,
            "entity_name": None,
            "tool_calls": calls,
        }

    # --- Fetch from RA GraphQL API ---
    try:
        raw = fetch_ra_entity(entity_type, identifier)
        normalized = normalize_ra_data(entity_type, raw)
        calls.append({
            "agent": "ra_scraper", "tool": "fetch_ra_entity",
            "args": {"entity_type": entity_type, "identifier": identifier},
            "result_fields": list(k for k, v in normalized.items() if v is not None),
        })
    except Exception as e:
        print(f"[ENRICHMENT] RA fetch failed: {e}")
        calls.append({"agent": "ra_scraper", "tool": "fetch_ra_entity", "error": str(e)})
        # Build minimal fallback from URL
        fallback_name = identifier.replace("-", " ").replace("_", " ").title()
        normalized = {"name": fallback_name, "page_type": entity_type}

    entity_name = normalized.get("name") or identifier

    return {
        "messages": [SystemMessage(content=f"RA data fetched for '{entity_name}' ({entity_type})")],
        "ra_entity_type": entity_type,
        "ra_identifier": identifier,
        "ra_data": normalized,
        "entity_name": entity_name,
        "tool_calls": calls,
    }


# =============================================================================
# NODE 2: GAP ANALYSIS
# =============================================================================


def gap_analysis_node(state: EnrichmentState) -> dict:
    """Examine RA data and identify which enrichment fields are missing."""
    ra_data = state.get("ra_data") or {}
    missing: List[str] = []

    if not ra_data.get("instagram"):
        missing.append("instagram")
    if not ra_data.get("soundcloud"):
        missing.append("soundcloud")
    if not ra_data.get("profile_picture"):
        missing.append("profile_picture")
    if not ra_data.get("bio"):
        missing.append("bio")

    return {
        "messages": [SystemMessage(content=f"Missing fields: {missing}")],
        "missing_fields": missing,
        "tool_calls": [{
            "agent": "gap_analysis", "tool": "analyze_gaps",
            "result": {"missing": missing, "total_ra_fields": sum(1 for v in ra_data.values() if v is not None)},
        }],
    }


def route_discovery_agents(state: EnrichmentState) -> List[str]:
    """Determine which discovery agents to dispatch based on missing fields."""
    missing = state.get("missing_fields") or []
    targets: List[str] = []
    if "instagram" in missing:
        targets.append("instagram_node")
    if "soundcloud" in missing:
        targets.append("soundcloud_node")
    # If nothing is missing, go straight to resolution
    return targets if targets else ["resolution_node"]


# =============================================================================
# NODE 3a: INSTAGRAM DISCOVERY
# =============================================================================


def instagram_discovery_node(state: EnrichmentState) -> dict:
    """Discover and validate an Instagram account for the entity."""
    entity_name = state.get("entity_name") or "Unknown"
    ra_data = state.get("ra_data") or {}
    calls: List[Dict[str, Any]] = []

    # --- Step 1: Search via HikerAPI ---
    results = search_instagram_accounts(entity_name)
    calls.append({
        "agent": "instagram_discovery", "tool": "hiker_search",
        "args": {"query": entity_name}, "result_count": len(results),
    })

    if not results:
        # Fallback: web search
        fallback_query = f"{entity_name} {ra_data.get('page_type', 'artist')} instagram official"
        search_result = _search_api(fallback_query)
        calls.append({"agent": "instagram_discovery", "tool": "web_search_fallback", "args": {"query": fallback_query}})
        if search_result:
            url = _extract_url_from_text(search_result, "instagram")
            if url and _validate_url(url):
                return {
                    "messages": [SystemMessage(content=f"Instagram found via web search: {url}")],
                    "instagram_result": {"url": url, "source": "web_search", "confidence": 0.6},
                    "tool_calls": calls,
                }
        return {
            "messages": [SystemMessage(content="Instagram not found")],
            "instagram_result": None,
            "tool_calls": calls,
        }

    # --- Step 2: LLM cross-reference to pick best match ---
    candidates = [
        {
            "username": r.get("username"),
            "full_name": r.get("full_name"),
            "biography": (r.get("biography") or "")[:200],
            "follower_count": r.get("follower_count"),
            "is_verified": r.get("is_verified", False),
        }
        for r in results[:5]
    ]

    ra_context = {
        "name": entity_name,
        "type": ra_data.get("page_type"),
        "bio": (ra_data.get("bio") or "")[:300],
        "country": ra_data.get("country"),
    }

    prompt = (
        "You are evaluating Instagram search results to find the correct account "
        "for an electronic music entity.\n\n"
        f"Entity from Resident Advisor:\n{json.dumps(ra_context, indent=2)}\n\n"
        f"Instagram search candidates:\n{json.dumps(candidates, indent=2)}\n\n"
        "Return a JSON object with:\n"
        '- "best_match_index": index (0-based) of the best matching account, or -1 if none match\n'
        '- "confidence": float 0.0 to 1.0\n'
        '- "reasoning": brief explanation\n\n'
        "Only match if you are reasonably confident this is the same entity."
    )

    with using_attributes(tags=["instagram_discovery", "enrichment"]):
        response = llm.invoke([
            SystemMessage(content="You are a data matching specialist for the electronic music industry. Respond with JSON only."),
            HumanMessage(content=prompt),
        ])

    calls.append({"agent": "instagram_discovery", "tool": "llm_cross_reference", "args": {"candidates": len(candidates)}})

    match_result = _parse_llm_json(response.content if response.content else "")
    idx = (match_result or {}).get("best_match_index", -1)
    confidence = (match_result or {}).get("confidence", 0.0)

    if isinstance(idx, int) and 0 <= idx < len(results) and confidence >= 0.6:
        matched = results[idx]
        user_id = str(matched.get("pk") or matched.get("id") or "")

        # Fetch detailed profile for high-res picture + full bio
        profile = get_instagram_profile(user_id) if user_id else None
        calls.append({"agent": "instagram_discovery", "tool": "hiker_profile", "args": {"user_id": user_id}})

        source_data = profile or matched
        return {
            "messages": [SystemMessage(content=f"Instagram matched: @{matched.get('username')} (confidence={confidence:.2f})")],
            "instagram_result": {
                "url": f"https://instagram.com/{matched.get('username')}",
                "profile_pic_url": source_data.get("hd_profile_pic_url_info", {}).get("url")
                    or source_data.get("profile_pic_url"),
                "biography": source_data.get("biography"),
                "follower_count": source_data.get("follower_count"),
                "source": "hiker_api",
                "confidence": confidence,
            },
            "tool_calls": calls,
        }

    return {
        "messages": [SystemMessage(content=f"No confident Instagram match (best confidence={confidence:.2f})")],
        "instagram_result": None,
        "tool_calls": calls,
    }


# =============================================================================
# NODE 3b: SOUNDCLOUD DISCOVERY
# =============================================================================


def soundcloud_discovery_node(state: EnrichmentState) -> dict:
    """Discover and validate a SoundCloud account for the entity."""
    entity_name = state.get("entity_name") or "Unknown"
    ra_data = state.get("ra_data") or {}
    calls: List[Dict[str, Any]] = []

    # --- Step 1: Search via SoundCloud API ---
    results = search_soundcloud_users(entity_name)
    calls.append({
        "agent": "soundcloud_discovery", "tool": "soundcloud_search",
        "args": {"query": entity_name}, "result_count": len(results),
    })

    if not results:
        # Fallback: web search
        fallback_query = f"{entity_name} soundcloud official"
        search_result = _search_api(fallback_query)
        calls.append({"agent": "soundcloud_discovery", "tool": "web_search_fallback", "args": {"query": fallback_query}})
        if search_result:
            url = _extract_url_from_text(search_result, "soundcloud")
            if url and _validate_url(url):
                return {
                    "messages": [SystemMessage(content=f"SoundCloud found via web search: {url}")],
                    "soundcloud_result": {"url": url, "source": "web_search", "confidence": 0.6},
                    "tool_calls": calls,
                }
        return {
            "messages": [SystemMessage(content="SoundCloud not found")],
            "soundcloud_result": None,
            "tool_calls": calls,
        }

    # --- Step 2: LLM cross-reference ---
    candidates = [
        {
            "permalink": r.get("permalink"),
            "username": r.get("username"),
            "full_name": r.get("full_name") or r.get("username"),
            "description": (r.get("description") or "")[:200],
            "followers_count": r.get("followers_count"),
        }
        for r in results[:5]
    ]

    ra_context = {
        "name": entity_name,
        "type": ra_data.get("page_type"),
        "bio": (ra_data.get("bio") or "")[:300],
        "country": ra_data.get("country"),
    }

    prompt = (
        "You are evaluating SoundCloud search results to find the correct account "
        "for an electronic music entity.\n\n"
        f"Entity from Resident Advisor:\n{json.dumps(ra_context, indent=2)}\n\n"
        f"SoundCloud search candidates:\n{json.dumps(candidates, indent=2)}\n\n"
        "Return a JSON object with:\n"
        '- "best_match_index": index (0-based) of the best matching account, or -1 if none match\n'
        '- "confidence": float 0.0 to 1.0\n'
        '- "reasoning": brief explanation\n\n'
        "Only match if you are reasonably confident this is the same entity."
    )

    with using_attributes(tags=["soundcloud_discovery", "enrichment"]):
        response = llm.invoke([
            SystemMessage(content="You are a data matching specialist for the electronic music industry. Respond with JSON only."),
            HumanMessage(content=prompt),
        ])

    calls.append({"agent": "soundcloud_discovery", "tool": "llm_cross_reference", "args": {"candidates": len(candidates)}})

    match_result = _parse_llm_json(response.content if response.content else "")
    idx = (match_result or {}).get("best_match_index", -1)
    confidence = (match_result or {}).get("confidence", 0.0)

    if isinstance(idx, int) and 0 <= idx < len(results) and confidence >= 0.6:
        matched = results[idx]
        permalink = matched.get("permalink") or matched.get("username") or ""
        return {
            "messages": [SystemMessage(content=f"SoundCloud matched: {permalink} (confidence={confidence:.2f})")],
            "soundcloud_result": {
                "url": f"https://soundcloud.com/{permalink}",
                "avatar_url": matched.get("avatar_url"),
                "description": matched.get("description"),
                "source": "soundcloud_api",
                "confidence": confidence,
            },
            "tool_calls": calls,
        }

    return {
        "messages": [SystemMessage(content=f"No confident SoundCloud match (best confidence={confidence:.2f})")],
        "soundcloud_result": None,
        "tool_calls": calls,
    }


# =============================================================================
# NODE 4: RESOLUTION (profile picture + bio priority cascade)
# =============================================================================


def resolution_node(state: EnrichmentState) -> dict:
    """Resolve profile picture and bio using priority-based cascades."""
    ra_data = state.get("ra_data") or {}
    ig = state.get("instagram_result") or {}
    sc = state.get("soundcloud_result") or {}
    entity_name = state.get("entity_name") or "Unknown"
    calls: List[Dict[str, Any]] = []

    # --- Profile picture: RA > Instagram > SoundCloud > Google Search ---
    profile_picture: Optional[Dict[str, Any]] = None

    if ra_data.get("profile_picture"):
        profile_picture = {
            "value": ra_data["profile_picture"],
            "source": "ra",
            "confidence": 0.9,
        }
    elif ig.get("profile_pic_url"):
        profile_picture = {
            "value": ig["profile_pic_url"],
            "source": "instagram",
            "confidence": ig.get("confidence", 0.8),
        }
    elif sc.get("avatar_url"):
        # Upgrade SoundCloud avatar to high-res if possible
        avatar = sc["avatar_url"]
        if avatar:
            avatar = avatar.replace("-large.", "-t500x500.")
        profile_picture = {
            "value": avatar,
            "source": "soundcloud",
            "confidence": sc.get("confidence", 0.7),
        }
    else:
        # Google Search fallback
        search_query = f"{entity_name} {ra_data.get('page_type', 'artist')} profile photo"
        search_result = _search_api(search_query)
        calls.append({"agent": "resolution", "tool": "google_image_fallback", "args": {"query": search_query}})
        if search_result:
            profile_picture = {
                "value": search_result,
                "source": "google_search",
                "confidence": 0.4,
            }

    # --- Bio: Instagram > SoundCloud > RA > LLM fallback ---
    bio: Optional[Dict[str, Any]] = None

    if ig.get("biography") and len(ig["biography"]) > 20:
        bio = {
            "value": ig["biography"],
            "source": "instagram",
            "confidence": ig.get("confidence", 0.8),
        }
    elif sc.get("description") and len(sc["description"]) > 20:
        bio = {
            "value": sc["description"],
            "source": "soundcloud",
            "confidence": sc.get("confidence", 0.7),
        }
    elif ra_data.get("bio") and len(ra_data["bio"]) > 20:
        bio = {
            "value": ra_data["bio"],
            "source": "ra",
            "confidence": 0.95,
        }
    else:
        # LLM fallback
        entity_type = ra_data.get("page_type", "artist")
        with using_attributes(tags=["resolution", "bio_generation"]):
            response = llm.invoke([
                SystemMessage(content="You are an electronic music journalist. Write a brief, factual biography."),
                HumanMessage(content=f"Write a 2-3 sentence biography for electronic music {entity_type} '{entity_name}'."),
            ])
        calls.append({"agent": "resolution", "tool": "llm_bio_generation"})
        if response.content:
            bio = {
                "value": _compact(response.content, limit=500),
                "source": "llm_generated",
                "confidence": 0.5,
            }

    calls.append({
        "agent": "resolution", "tool": "priority_cascade",
        "result": {
            "pic_source": (profile_picture or {}).get("source"),
            "bio_source": (bio or {}).get("source"),
        },
    })

    return {
        "messages": [SystemMessage(content="Resolution complete")],
        "profile_picture": profile_picture,
        "bio": bio,
        "tool_calls": calls,
    }


# =============================================================================
# NODE 5: ASSEMBLY
# =============================================================================


def assembly_node(state: EnrichmentState) -> dict:
    """Assemble the final enriched profile from all sources."""
    ra_data = state.get("ra_data") or {}
    ig = state.get("instagram_result") or {}
    sc = state.get("soundcloud_result") or {}
    profile_picture = state.get("profile_picture")
    bio = state.get("bio")
    calls: List[Dict[str, Any]] = []

    final_profile: Dict[str, Any] = {}

    # --- RA base fields (confidence 1.0 — structured data) ---
    ra_base_fields = [
        "name", "page_type", "address", "latitude", "longitude",
        "website", "facebook", "twitter", "bandcamp", "discogs",
        "country", "resident_country", "city", "capacity",
        "follower_count", "content_url", "cover_picture",
    ]
    for field in ra_base_fields:
        val = ra_data.get(field)
        if val is not None:
            final_profile[field] = {"value": val, "source": "ra", "confidence": 1.0}

    # --- Enriched: profile_picture ---
    if profile_picture:
        final_profile["profile_picture"] = profile_picture

    # --- Enriched: biography ---
    if bio:
        final_profile["biography"] = bio

    # --- Enriched: Instagram ---
    if ig.get("url"):
        final_profile["instagram"] = {
            "value": ig["url"],
            "source": ig.get("source", "hiker_api"),
            "confidence": ig.get("confidence", 0.8),
        }
    elif ra_data.get("instagram"):
        final_profile["instagram"] = {
            "value": ra_data["instagram"],
            "source": "ra",
            "confidence": 1.0,
        }

    # --- Enriched: SoundCloud ---
    if sc.get("url"):
        final_profile["soundcloud"] = {
            "value": sc["url"],
            "source": sc.get("source", "soundcloud_api"),
            "confidence": sc.get("confidence", 0.8),
        }
    elif ra_data.get("soundcloud"):
        final_profile["soundcloud"] = {
            "value": ra_data["soundcloud"],
            "source": "ra",
            "confidence": 1.0,
        }

    # --- LLM consistency review ---
    # Build a compact summary for the LLM to review
    review_summary = {
        k: {"value": str(v.get("value", ""))[:100], "source": v.get("source"), "confidence": v.get("confidence")}
        for k, v in final_profile.items()
        if isinstance(v, dict)
    }

    review_prompt = (
        f"Review this enriched profile for '{state.get('entity_name')}' "
        f"({ra_data.get('page_type', 'entity')}) and check for consistency.\n\n"
        f"Profile:\n{json.dumps(review_summary, indent=2)}\n\n"
        "If any fields look suspicious or contradictory, return a JSON object with:\n"
        '- "adjustments": list of {{"field": str, "new_confidence": float, "reason": str}}\n'
        '- "review_notes": brief summary\n\n'
        "If everything looks consistent, return: {\"adjustments\": [], \"review_notes\": \"All fields consistent\"}"
    )

    with using_attributes(tags=["assembly", "enrichment"]):
        response = llm.invoke([
            SystemMessage(content="You are a data quality specialist for electronic music entities. Respond with JSON only."),
            HumanMessage(content=review_prompt),
        ])

    calls.append({"agent": "assembly", "tool": "llm_consistency_review"})

    review = _parse_llm_json(response.content if response.content else "")
    if review and review.get("adjustments"):
        for adj in review["adjustments"]:
            field = adj.get("field")
            new_conf = adj.get("new_confidence")
            if field and field in final_profile and isinstance(new_conf, (int, float)):
                final_profile[field]["confidence"] = new_conf
                print(f"[ENRICHMENT] Confidence adjusted: {field} -> {new_conf} ({adj.get('reason')})")

    calls.append({
        "agent": "assembly", "tool": "build_profile",
        "result": {"field_count": len(final_profile), "review_notes": (review or {}).get("review_notes")},
    })

    return {
        "messages": [SystemMessage(content=f"Assembly complete: {len(final_profile)} fields")],
        "final_profile": final_profile,
        "tool_calls": calls,
    }


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def build_enrichment_graph():
    """Build the LangGraph workflow for RA entity enrichment."""
    g = StateGraph(EnrichmentState)

    g.add_node("ra_scraper_node", ra_scraper_node)
    g.add_node("gap_analysis_node", gap_analysis_node)
    g.add_node("instagram_node", instagram_discovery_node)
    g.add_node("soundcloud_node", soundcloud_discovery_node)
    g.add_node("resolution_node", resolution_node)
    g.add_node("assembly_node", assembly_node)

    # Sequential: START -> RA scrape -> gap analysis
    g.add_edge(START, "ra_scraper_node")
    g.add_edge("ra_scraper_node", "gap_analysis_node")

    # Conditional parallel: gap analysis routes to discovery agents or straight to resolution
    g.add_conditional_edges(
        "gap_analysis_node",
        route_discovery_agents,
        {
            "instagram_node": "instagram_node",
            "soundcloud_node": "soundcloud_node",
            "resolution_node": "resolution_node",
        },
    )

    # Discovery agents converge to resolution
    g.add_edge("instagram_node", "resolution_node")
    g.add_edge("soundcloud_node", "resolution_node")

    # Resolution -> Assembly -> END
    g.add_edge("resolution_node", "assembly_node")
    g.add_edge("assembly_node", END)

    return g.compile()
