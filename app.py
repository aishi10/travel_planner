from __future__ import annotations

import json
import os
import html
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import requests
import streamlit as st


USER_AGENT = "free-travel-planner/1.0 (educational project)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
GEMINI_MODEL = "gemini-2.5-flash"


INTEREST_SUGGESTIONS = {
    "food": ["local market", "food tour", "well-reviewed restaurant", "cafe district"],
    "history": ["museum", "old town walk", "heritage site", "guided walking tour"],
    "nature": ["park", "day trip", "viewpoint", "nature reserve"],
    "nightlife": ["rooftop bar", "live music venue", "night market", "central nightlife district"],
    "shopping": ["local bazaar", "shopping street", "mall", "artisan market"],
    "art": ["gallery", "street art area", "design district", "museum"],
    "relaxation": ["spa", "beach", "scenic park", "slow cafe"],
    "adventure": ["hike", "water sport", "day excursion", "outdoor activity"],
}

HOTEL_STYLE = {
    "budget": "simple guesthouses, hostels, and 3-star stays",
    "mid-range": "comfortable 3 to 4 star hotels in a central area",
    "luxury": "well-rated 4 to 5 star hotels with strong location and amenities",
}

WIKI_ATTRACTION_ALLOW_KEYWORDS = (
    "temple",
    "beach",
    "museum",
    "park",
    "garden",
    "palace",
    "market",
    "waterfall",
    "viewpoint",
    "monument",
    "cathedral",
    "church",
    "mosque",
    "shrine",
    "fort",
    "castle",
    "lake",
    "island",
    "zoo",
    "aquarium",
    "gallery",
    "square",
    "promenade",
    "harbor",
    "harbour",
    "cliff",
    "rice terrace",
    "rice terraces",
    "walk",
    "trail",
    "old town",
    "heritage",
    "landmark",
)

WIKI_ATTRACTION_BLOCK_KEYWORDS = (
    "history",
    "economy",
    "demography",
    "politics",
    "bombing",
    "bombings",
    "war",
    "attack",
    "incident",
    "municipality",
    "province",
    "district",
    "regency",
    "people",
    "language",
    "culture",
    "geography",
    "transport",
    "education",
    "list of",
    "timeline",
    "battle",
    "administration",
    "climate",
)


@dataclass(slots=True)
class Place:
    name: str
    kind: str
    distance_km: float | None = None
    lat: float | None = None
    lon: float | None = None
    note: str | None = None


def http_get(url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, timeout: int = 20):
    merged_headers = {"User-Agent": USER_AGENT}
    if headers:
        merged_headers.update(headers)
    return requests.get(url, params=params, headers=merged_headers, timeout=timeout)


def http_post(url: str, data: str, timeout: int = 30):
    return requests.post(url, data=data, headers={"User-Agent": USER_AGENT, "Content-Type": "application/x-www-form-urlencoded"}, timeout=timeout)


def safe_json_loads(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        match = None
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                try:
                    return json.loads(candidate)
                except Exception:
                    continue
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
        return None


def clean_snippet(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def get_gemini_client():
    try:
        from google import genai

        if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            return None
        return genai.Client()
    except Exception:
        return None


def gemini_is_available() -> bool:
    return get_gemini_client() is not None


def geocode_destination(destination: str) -> dict[str, Any] | None:
    try:
        response = http_get(
            NOMINATIM_URL,
            params={"q": destination, "format": "jsonv2", "limit": 1, "addressdetails": 1},
        )
        response.raise_for_status()
        items = response.json()
        return items[0] if items else None
    except Exception:
        return None


def overpass_query(lat: float, lon: float, radius_m: int, tags: list[tuple[str, str]], limit: int = 12) -> list[Place]:
    tag_filters = "".join([f'nwr(around:{radius_m},{lat},{lon})["{key}"="{value}"];' for key, value in tags])
    query = f"""
    [out:json][timeout:25];
    (
      {tag_filters}
    );
    out center;
    """
    try:
        response = http_post(OVERPASS_URL, data=query.strip())
        response.raise_for_status()
        elements = response.json().get("elements", [])
    except Exception:
        return []

    places: list[Place] = []
    for item in elements:
        tags_data = item.get("tags", {})
        name = tags_data.get("name")
        if not name:
            continue
        item_lat = item.get("lat", item.get("center", {}).get("lat"))
        item_lon = item.get("lon", item.get("center", {}).get("lon"))
        distance = haversine_km(lat, lon, item_lat, item_lon) if item_lat and item_lon else None
        note = tags_data.get("cuisine") or tags_data.get("brand") or tags_data.get("website")
        kind = next((f"{key}={value}" for key, value in tags if tags_data.get(key) == value), "place")
        places.append(Place(name=name, kind=kind, distance_km=distance, lat=item_lat, lon=item_lon, note=note))

    places.sort(key=lambda item: item.distance_km if item.distance_km is not None else 9999)
    return places[:limit]


def open_meteo_forecast(lat: float, lon: float) -> dict[str, Any] | None:
    try:
        response = http_get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode",
                "timezone": "auto",
            },
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def wikipedia_attractions(destination: str, interests: list[str], limit: int = 6) -> list[Place]:
    queries = [
        f"{destination} attractions",
        f"best things to do in {destination}",
        f"{destination} landmarks",
    ] + [f"{destination} {interest}" for interest in interests[:3]]
    results: list[Place] = []
    seen: set[str] = set()
    destination_key = destination.strip().lower()

    def looks_like_attraction_title(title: str) -> bool:
        normalized = title.strip().lower()
        if not normalized or normalized == destination_key:
            return False
        if destination_key and normalized.startswith(destination_key) and len(normalized) <= len(destination_key) + 3:
            return False
        if any(block in normalized for block in WIKI_ATTRACTION_BLOCK_KEYWORDS):
            return False
        return any(keyword in normalized for keyword in WIKI_ATTRACTION_ALLOW_KEYWORDS)

    for query in queries:
        try:
            response = http_get(
                WIKIPEDIA_SEARCH_URL,
                params={"q": query, "limit": limit},
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            pages = response.json().get("pages", [])
        except Exception:
            continue

        for page in pages:
            title = page.get("title")
            if not title:
                continue
            if not looks_like_attraction_title(title):
                continue
            normalized = title.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            summary = clean_snippet(page.get("excerpt") or page.get("description"))
            results.append(Place(name=title.strip(), kind="wiki_attraction", note=summary[:160] if summary else None))
            if len(results) >= limit:
                return results[:limit]

    return results[:limit]


def candidate_snapshot(places: list[Place]) -> list[dict[str, Any]]:
    snapshot = []
    for place in places[:10]:
        snapshot.append(
            {
                "name": place.name,
                "kind": place.kind,
                "distance_km": round(place.distance_km, 2) if place.distance_km is not None else None,
                "note": place.note,
            }
        )
    return snapshot


def refine_with_gemini(plan: dict[str, Any], travel_dates: list[date], interests: list[str], budget: str, travelers: int) -> dict[str, Any] | None:
    client = get_gemini_client()
    if client is None:
        return None

    prompt = f"""
You are an expert travel planner. Use the free API results and user preferences below to create a personalized itinerary.
Return only valid JSON with these keys:
summary, itinerary_days, hotels, restaurants, attractions, budget, visa, season, tips, avoid, alternatives.

Rules:
- Always tailor the plan to the destination and interests.
- Mention what to avoid.
- Keep morning/afternoon/evening structure.
- Prefer the provided place candidate names for hotels and restaurants.
- For attractions, prefer the provided candidates, but you may also use well-known real landmarks for the destination or Wikipedia-sourced attractions when the free data is sparse.
- Do not invent hotels or restaurants.
- If a suggestion is not from the candidate lists, mark it as lower confidence and add a short verify-before-booking note.
- Use concise, practical wording.

User preferences:
{json.dumps({
    "destination": plan.get("destination_label"),
    "departure_city": plan.get("departure_city"),
    "travel_dates": [d.isoformat() for d in travel_dates],
    "duration_days": len(travel_dates),
    "budget": budget,
    "travelers": travelers,
    "interests": interests,
}, indent=2)}

Free API candidates:
{json.dumps({
    "hotel_names": plan.get("hotel_candidates", []),
    "restaurant_names": plan.get("restaurant_candidates", []),
    "attraction_names": plan.get("attraction_candidates", []),
    "wiki_attraction_names": plan.get("wiki_attractions", []),
    "weather": plan.get("weather", {}),
}, indent=2)}

Current local plan:
{json.dumps({
    "summary": plan.get("summary"),
    "itinerary_days": plan.get("itinerary_days"),
    "budget": plan.get("budget"),
    "visa": plan.get("visa"),
    "season": plan.get("season"),
    "tips": plan.get("tips"),
}, indent=2)}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.5,
                "response_mime_type": "application/json",
            },
        )
        text = getattr(response, "text", "") or ""
        parsed = safe_json_loads(text)
        return parsed
    except Exception:
        return None


def haversine_km(lat1: float, lon1: float, lat2: float | None, lon2: float | None) -> float | None:
    if lat2 is None or lon2 is None:
        return None
    from math import asin, cos, radians, sin, sqrt

    r = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def trip_style_summary(budget: str, interests: list[str]) -> str:
    vibe = ", ".join(interests[:3]) if interests else "a balanced mix of sights"
    return (
        f"This trip is planned as a {budget} experience with emphasis on {vibe}. "
        f"The goal is to keep the pace comfortable, use the destination efficiently, and match the day plan to your priorities."
    )


def budget_style_text(budget: str) -> str:
    return {
        "budget": "budget-friendly with simple, practical choices and fewer splurges",
        "mid-range": "comfortable with a balance of convenience and value",
        "luxury": "high-comfort with premium locations, service, and flexibility",
    }[budget]


def destination_cost_multiplier(destination: str) -> float:
    text = destination.lower()
    multipliers = {
        "tokyo": 1.6,
        "japan": 1.5,
        "london": 1.7,
        "paris": 1.5,
        "new york": 1.8,
        "new york city": 1.8,
        "dubai": 1.6,
        "singapore": 1.55,
        "bali": 0.6,
        "thailand": 0.55,
        "india": 0.45,
        "vietnam": 0.5,
        "indonesia": 0.55,
        "malaysia": 0.65,
    }
    for key, value in multipliers.items():
        if key in text:
            return value
    return 1.0


def budget_estimate(destination: str, duration: int, travelers: int, budget: str) -> dict[str, str]:
    per_day = {"budget": 75, "mid-range": 180, "luxury": 420}[budget] * destination_cost_multiplier(destination)
    stay_share = {"budget": 0.45, "mid-range": 0.5, "luxury": 0.6}[budget]
    food_share = {"budget": 0.25, "mid-range": 0.22, "luxury": 0.18}[budget]
    act_share = 0.18
    hotel_share = stay_share
    if travelers > 2:
        hotel_share *= 0.9

    total = per_day * duration * travelers
    flights = total * 0.35
    stay = total * hotel_share
    food = total * food_share
    activities = total * act_share

    return {
        "flights": f"${flights:,.0f}",
        "stay": f"${stay:,.0f}",
        "food": f"${food:,.0f}",
        "activities": f"${activities:,.0f}",
        "total": f"${(flights + stay + food + activities):,.0f}",
    }


def visa_tip(destination: str) -> str:
    text = destination.lower()
    if any(name in text for name in ["japan", "schengen", "uk", "united kingdom", "canada", "usa", "australia"]):
        return "Check visa and entry rules early, because requirements can change by nationality."
    return "Verify visa, passport validity, and entry rules for your nationality before booking."


def season_tip(destination: str, travel_dates: list[date]) -> str:
    if not travel_dates:
        return "Check local weather and seasonal patterns before booking."
    month = travel_dates[0].month
    if month in {6, 7, 8}:
        return "Expect peak summer demand and higher prices; book popular stays early."
    if month in {12, 1, 2}:
        return "Winter conditions may affect transport and activities in some destinations."
    return "This is often a balanced travel period, but local events can still move prices."


def local_tips(destination: str) -> list[str]:
    return [
        f"Compare hotels in {destination} by neighborhood, not just by price. A central area usually saves more money than a cheap room far from the sights.",
        f"Avoid accepting transport or restaurant offers from people approaching you around major tourist areas in {destination}.",
        "Use licensed taxis or known ride apps where available.",
        "Watch for common tourist scams near stations, monuments, and airport transfer desks.",
    ]


def choose_focus_places(places: list[Place], label: str, fallback: list[str], include_note: bool = False) -> list[str]:
    if places:
        selected = []
        for place in places[:3]:
            dist = f" ({place.distance_km:.1f} km away)" if place.distance_km is not None else ""
            extra = f" - {place.note}" if include_note and place.note else ""
            selected.append(f"{place.name}{dist}{extra}")
        return selected
    return fallback[:3]


def _place_detail(place: str, destination: str, budget: str, interest: str, kind: str, distance_km: float | None = None, note: str | None = None, confidence: str = "medium") -> str:
    budget_desc = budget_style_text(budget)
    distance_text = f" about {distance_km:.1f} km from the center" if distance_km is not None else ""
    if note:
        note_text = f" Cuisine or venue note: {note}."
    else:
        note_text = ""
    confidence_text = {
        "high": "High-confidence pick.",
        "medium": "Reasonable pick to verify before booking.",
        "low": "Use as a backup and verify details before you commit.",
    }[confidence]

    if kind == "hotel":
        return (
            f"{place}: a {budget_desc} base in {destination}{distance_text}. "
            f"Good if you want to keep transit simple, return easily after dinner, and stay close to the main action. "
            f"{confidence_text}{note_text}"
        )
    if kind == "restaurant":
        return (
            f"{place}: a good match for a {interest}-focused meal in {destination}{distance_text}. "
            f"It suits a {budget_desc} trip, especially for lunch or dinner when you want a local experience without overcomplicating the day. "
            f"{confidence_text}{note_text}"
        )
    if kind == "attraction":
        return (
            f"{place}: a strong {interest} stop in {destination}. "
            f"Plan this for the morning or afternoon, then pair it with a meal nearby so the day feels balanced and easy to follow. "
            f"{confidence_text}{note_text}"
        )
    return place


def detail_for_slot(kind: str, destination: str, interest: str, item: str, part_of_day: str) -> str:
    base = item.strip()
    if not base:
        base = destination

    if kind == "morning":
        return (
            f"Begin with {base} to ease into {destination}. "
            f"This gives you a low-pressure start and a chance to get oriented before the busier part of the day."
        )
    if kind == "afternoon":
        return (
            f"Spend the afternoon around {base}, which fits well with a {interest} focus in {destination}. "
            f"This is a good time for a longer visit, lunch, or a slower-paced experience that matches your budget."
        )
    if kind == "evening":
        return (
            f"In the evening, head to {base} for a relaxed {interest} finish in {destination}. "
            f"Keep the pace lighter so you can enjoy dinner, avoid overspending, and recover for the next day."
        )
    if kind == "hotel":
        return (
            f"Stay at {base}. Choosing a centrally located base in {destination} keeps transit simple, reduces taxi costs, and makes it easier to return after dinner."
        )
    return base


def hotel_area_hint(destination: str) -> str:
    text = destination.lower()
    hints = {
        "tokyo": "Shibuya, Shinjuku, or Ginza",
        "japan": "a central rail-linked area",
        "paris": "the 1st, 3rd, 7th, or 9th arrondissement",
        "london": "Covent Garden, Westminster, South Bank, or Kensington",
        "new york": "Midtown, Chelsea, or Lower Manhattan",
        "new york city": "Midtown, Chelsea, or Lower Manhattan",
        "bali": "Seminyak, Ubud, Sanur, or Canggu depending on your plans",
        "dubai": "Downtown Dubai, Marina, or Jumeirah",
        "singapore": "Marina Bay, Orchard, or Clarke Quay",
        "thailand": "a central district close to transit or nightlife",
        "vietnam": "a walkable central district near the old quarter or riverfront",
    }
    for key, hint in hints.items():
        if key in text:
            return hint
    return "a central, walkable area close to transit and dining"


def hotel_recommendations(places: list[Place], destination: str, budget: str) -> list[str]:
    area_hint = hotel_area_hint(destination)
    if places:
        recommendations: list[str] = []
        for place in places[:4]:
            distance_text = f"{place.distance_km:.1f} km from the center" if place.distance_km is not None else "a central area"
            note_text = f" It lists {place.note}." if place.note else ""
            recommendations.append(
                (
                    f"{place.name}: a {budget_style_text(budget)} stay in {destination}, about {distance_text}. "
                    f"Best if you want to base yourself near {area_hint} so sightseeing, meals, and evening returns stay simple. "
                    f"Look for easy taxi or rail access and verify the exact neighborhood before booking.{note_text}"
                )
            )
        return recommendations
    return [
        f"Look for {HOTEL_STYLE[budget]} in {destination}. A strong choice is usually in or near {area_hint}, because a central location makes it easier to reach sightseeing, food streets, and evening plans without extra transit.",
    ]


def restaurant_recommendations(places: list[Place], destination: str, budget: str, interests: list[str]) -> list[str]:
    focus = interests[0] if interests else "food"
    if places:
        return [
            _place_detail(
                place.name,
                destination,
                budget,
                focus,
                "restaurant",
                place.distance_km,
                place.note,
                confidence="high" if place.note else "medium",
            )
            for place in places[:4]
        ]
    return [
        f"Choose a well-reviewed {focus} restaurant in {destination}. For a {budget_style_text(budget)} trip, aim for a place that is central, easy to reach, and good for either lunch or an early dinner.",
    ]


def attraction_recommendations(places: list[Place], destination: str, budget: str, interests: list[str]) -> list[str]:
    focus = interests[0] if interests else "sightseeing"
    if places:
        return [
            _place_detail(place.name, destination, budget, focus, "attraction", place.distance_km, place.note)
            for place in places[:3]
        ]
    return [
        f"Explore the main highlights of {destination}. For a {budget} trip, keep one anchor attraction, one lighter walk, and one flexible backup in case the weather or crowds change your plans.",
    ]


def travel_tips(destination: str, budget: str, interests: list[str]) -> list[str]:
    focus = interests[0] if interests else "travel"
    tips = [
        f"For a {budget} trip in {destination}, keep your hotel central so you spend less on taxis and more on activities.",
        f"If your focus is {focus}, group nearby sights and meals together to avoid crossing the city multiple times.",
        f"Avoid tourist-heavy areas at peak meal times unless you already have a reservation and know the price range.",
    ]
    if budget == "budget":
        tips.append("Use local transport, eat lunch at smaller neighborhood spots, and book only the key paid activities you really care about.")
    elif budget == "mid-range":
        tips.append("A mid-range plan works best when you pay a bit more for location and save money by keeping activities clustered.")
    else:
        tips.append("For luxury travel, reserve the best hotel location first, then add premium dining or private transfers where they add real value.")
    tips.extend(local_tips(destination))
    return tips


def nearby_attractions(lat: float, lon: float, limit: int = 12) -> list[Place]:
    tags = [
        ("tourism", "attraction"),
        ("tourism", "museum"),
        ("tourism", "gallery"),
        ("tourism", "viewpoint"),
        ("tourism", "theme_park"),
        ("historic", "yes"),
        ("historic", "monument"),
        ("historic", "memorial"),
        ("leisure", "park"),
        ("natural", "peak"),
        ("amenity", "place_of_worship"),
        ("amenity", "theatre"),
    ]
    places = overpass_query(lat, lon, 8000, tags, limit=limit)
    cleaned: list[Place] = []
    seen: set[str] = set()
    for place in places:
        name = place.name.strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        cleaned.append(place)
    return cleaned[:limit]


def nearby_restaurants(lat: float, lon: float, limit: int = 15) -> list[Place]:
    tags = [
        ("amenity", "restaurant"),
        ("amenity", "cafe"),
        ("amenity", "fast_food"),
        ("amenity", "food_court"),
        ("amenity", "ice_cream"),
        ("amenity", "bar"),
        ("amenity", "pub"),
        ("amenity", "biergarten"),
        ("shop", "bakery"),
        ("shop", "deli"),
    ]
    places = overpass_query(lat, lon, 6000, tags, limit=limit)
    cleaned: list[Place] = []
    seen: set[str] = set()
    for place in places:
        name = place.name.strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        cleaned.append(place)
    return cleaned[:limit]


def build_itinerary(
    destination: str,
    travel_dates: list[date],
    interests: list[str],
    budget: str,
    travelers: int,
    departure_city: str,
) -> dict[str, Any]:
    geo = geocode_destination(destination)
    lat = float(geo["lat"]) if geo else None
    lon = float(geo["lon"]) if geo else None
    display_name = geo.get("display_name", destination) if geo else destination

    hotels = overpass_query(lat, lon, 3000, [("tourism", "hotel"), ("tourism", "guest_house"), ("tourism", "hostel"), ("tourism", "motel"), ("tourism", "resort"), ("tourism", "apartment")], limit=12) if lat and lon else []
    restaurants = nearby_restaurants(lat, lon, limit=15) if lat and lon else []
    wiki_attractions = wikipedia_attractions(destination, interests, limit=8)
    attractions = nearby_attractions(lat, lon, limit=12) if lat and lon else []
    attraction_pool = []
    seen_attractions: set[str] = set()
    for place in wiki_attractions + attractions:
        normalized = place.name.strip().lower()
        if normalized and normalized not in seen_attractions:
            seen_attractions.add(normalized)
            attraction_pool.append(place)

    weather = open_meteo_forecast(lat, lon) if lat and lon else None
    trip_days = max(1, (travel_dates[-1] - travel_dates[0]).days + 1) if travel_dates else 3
    selected_days = trip_days

    itinerary_days = []
    interest_pool = interests or ["culture", "food", "nature"]
    for idx in range(selected_days):
        day_focus = interest_pool[idx % len(interest_pool)]
        is_first_day = idx == 0
        is_last_day = idx == selected_days - 1
        attraction_names = choose_focus_places(
            attraction_pool[idx * 2 : idx * 2 + 2],
            "attraction",
            [
                f"a relaxed walking route through the center of {destination}",
                f"a local {day_focus} experience in {destination}",
            ],
        )
        restaurant_names = choose_focus_places(
            restaurants[idx * 2 : idx * 2 + 2],
            "restaurant",
            [f"a well-reviewed {day_focus} restaurant in {destination}"],
            include_note=True,
        )
        hotel_names = choose_focus_places(
            hotels[:3],
            "hotel",
            [f"a central {budget} stay in {destination}"],
        )
        if is_first_day:
            morning_text = f"Start the trip with {attraction_names[0]} so you can orient yourself in {destination} without overloading the day."
            afternoon_text = f"Use the afternoon for {attraction_names[1] if len(attraction_names) > 1 else attraction_names[0]}, then match it with a simple lunch near the same area."
            evening_text = f"Keep the first evening easy with {restaurant_names[0]} and an early return to your hotel."
        elif is_last_day:
            morning_text = f"Use the final morning for {attraction_names[0]} as a last easy highlight before departure or a relaxed wrap-up."
            afternoon_text = f"Leave the afternoon open for {attraction_names[1] if len(attraction_names) > 1 else attraction_names[0]} so you can finish the trip at a comfortable pace."
            evening_text = f"End the trip with {restaurant_names[0]} and keep the schedule light so checkout or travel is simple."
        else:
            morning_text = f"Begin with {attraction_names[0]} to keep the pace steady and stay close to the day's main area."
            afternoon_text = f"After lunch, spend time at {attraction_names[1] if len(attraction_names) > 1 else attraction_names[0]} for the most useful part of the {day_focus} plan."
            evening_text = f"Wrap up with {restaurant_names[0]} so the evening stays convenient and budget-aware."

        itinerary_days.append(
            {
                "day": idx + 1,
                "focus": day_focus,
                "morning": morning_text,
                "afternoon": afternoon_text,
                "evening": evening_text,
                "hotel_note": detail_for_slot("hotel", destination, day_focus, hotel_names[0], "hotel"),
            }
        )

    return {
        "destination_label": display_name,
        "geo": geo,
        "weather": weather,
        "itinerary_days": itinerary_days,
        "hotel_candidates": [place.name for place in hotels[:12]],
        "restaurant_candidates": [place.name for place in restaurants[:12]],
        "attraction_candidates": [place.name for place in attraction_pool[:12]],
        "wiki_attractions": [place.name for place in wiki_attractions[:12]],
        "hotels": hotel_recommendations(hotels, destination, budget),
        "restaurants": restaurant_recommendations(restaurants, destination, budget, interests),
        "attractions": attraction_recommendations(attraction_pool, destination, budget, interests),
        "budget": budget_estimate(destination, trip_days, travelers, budget),
        "visa": visa_tip(destination),
        "season": season_tip(destination, travel_dates),
        "tips": travel_tips(destination, budget, interests),
        "summary": trip_style_summary(budget, interests),
        "departure_city": departure_city,
    }


def normalize_gemini_plan(local_plan: dict[str, Any], gemini_plan: dict[str, Any] | None) -> dict[str, Any]:
    if not gemini_plan:
        return local_plan

    merged = dict(local_plan)
    for key in ["summary", "itinerary_days", "budget", "visa", "season", "tips", "avoid", "alternatives"]:
        if key in gemini_plan and gemini_plan[key]:
            merged[key] = gemini_plan[key]

    hotel_allowed = _candidate_names([Place(name=name, kind="hotel") for name in local_plan.get("hotel_candidates", [])])
    restaurant_allowed = _candidate_names([Place(name=name, kind="restaurant") for name in local_plan.get("restaurant_candidates", [])])
    attraction_allowed = _candidate_names([Place(name=name, kind="attraction") for name in local_plan.get("attraction_candidates", [])])
    attraction_allowed |= _candidate_names([Place(name=name, kind="wiki_attraction") for name in local_plan.get("wiki_attractions", [])])

    if gemini_plan.get("hotels"):
        merged["hotels"] = _filter_plan_list(gemini_plan.get("hotels"), hotel_allowed, local_plan.get("hotels", []))
    if gemini_plan.get("restaurants"):
        merged["restaurants"] = _filter_plan_list(gemini_plan.get("restaurants"), restaurant_allowed, local_plan.get("restaurants", []))
    if gemini_plan.get("attractions"):
        merged["attractions"] = _filter_plan_list(gemini_plan.get("attractions"), attraction_allowed, local_plan.get("attractions", []))
    return merged


def _candidate_names(places: list[Place]) -> set[str]:
    return {place.name.strip().lower() for place in places if place.name}


def _filter_plan_list(value: Any, allowed_names: set[str], fallback: list[str]) -> list[str]:
    items = []
    for item in as_list(value):
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("title") or item.get("place") or "").strip()
            detail = str(item.get("reason") or item.get("note") or item.get("description") or "").strip()
            normalized = re.split(r"\s*[-(]\s*", name.lower(), maxsplit=1)[0].strip()
            if normalized in allowed_names and name:
                items.append(f"{name} - {detail}" if detail else name)
        else:
            text = str(item).strip()
            normalized = re.split(r"\s*[-(]\s*", text.lower(), maxsplit=1)[0].strip()
            if normalized in allowed_names and text:
                items.append(text)
    return items if items else fallback


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def render_items(value: Any) -> list[str]:
    lines: list[str] = []
    for item in as_list(value):
        if isinstance(item, dict):
            name = item.get("name") or item.get("title") or item.get("place") or "Item"
            detail = item.get("reason") or item.get("note") or item.get("description")
            if detail:
                lines.append(f"{name} - {detail}")
            else:
                lines.append(str(name))
        else:
            lines.append(str(item))
    return lines


def inject_travel_branding() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

        :root {
            --bg-0: #fbf7f0;
            --bg-1: #f2ebe1;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-strong: rgba(255, 255, 255, 0.92);
            --gold: #9f7734;
            --gold-soft: rgba(159, 119, 52, 0.16);
            --cream: #1e262f;
            --muted: rgba(30, 38, 47, 0.72);
            --line: rgba(30, 38, 47, 0.1);
            --coral: #d66d5d;
            --shadow: 0 24px 60px rgba(30, 38, 47, 0.12);
        }

        html, body, [class*="css"] {
            font-family: "Space Grotesk", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(159, 119, 52, 0.12), transparent 28%),
                radial-gradient(circle at 85% 15%, rgba(214, 109, 93, 0.12), transparent 26%),
                radial-gradient(circle at 50% 120%, rgba(104, 140, 121, 0.12), transparent 30%),
                linear-gradient(145deg, var(--bg-0), var(--bg-1) 55%, #f8f3ea 100%);
            color: var(--cream);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 3.2rem;
        }

        h1, h2, h3, h4, h5 {
            font-family: "Cormorant Garamond", serif !important;
            letter-spacing: 0.02em;
            color: var(--cream) !important;
        }

        h1 {
            font-size: clamp(3.1rem, 6vw, 5.6rem) !important;
            line-height: 0.92 !important;
            margin-bottom: 0.25rem !important;
        }

        h2 {
            font-size: clamp(2rem, 3vw, 2.8rem) !important;
            margin-top: 0.8rem !important;
        }

        h3 {
            font-size: clamp(1.55rem, 2vw, 2.1rem) !important;
        }

        p, li, label, input, textarea, select, div, span {
            color: var(--cream);
        }

        .travel-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 30px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.78)),
                linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(247, 239, 227, 0.92));
            box-shadow: var(--shadow);
            padding: 1.35rem 1.45rem 1.1rem;
            margin: 0.2rem 0 1rem;
        }

        .travel-hero::before,
        .travel-hero::after {
            content: "";
            position: absolute;
            inset: auto;
            border-radius: 999px;
            filter: blur(4px);
            pointer-events: none;
        }

        .travel-hero::before {
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(159, 119, 52, 0.16), transparent 68%);
            top: -80px;
            right: -50px;
        }

        .travel-hero::after {
            width: 180px;
            height: 180px;
            background: radial-gradient(circle, rgba(214, 109, 93, 0.14), transparent 68%);
            bottom: -80px;
            left: -50px;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: 1.25fr 0.75fr;
            gap: 1rem;
            align-items: stretch;
            position: relative;
            z-index: 1;
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            font-size: 0.73rem;
            color: rgba(30, 38, 47, 0.68);
            margin-bottom: 0.9rem;
        }

        .hero-kicker::before {
            content: "";
            width: 34px;
            height: 1px;
            background: linear-gradient(90deg, transparent, var(--gold));
        }

        .hero-copy p {
            color: var(--muted);
            max-width: 64ch;
            font-size: 1.02rem;
            line-height: 1.72;
        }

        .destination-cycle {
            position: relative;
            display: inline-block;
            min-width: 10ch;
            margin-left: 0.15em;
            color: var(--gold);
            text-shadow: 0 0 18px rgba(159, 119, 52, 0.12);
        }

        .destination-cycle span {
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
            animation: cycleWords 12s infinite;
            white-space: nowrap;
        }

        .destination-cycle span:nth-child(1) { animation-delay: 0s; }
        .destination-cycle span:nth-child(2) { animation-delay: 3s; }
        .destination-cycle span:nth-child(3) { animation-delay: 6s; }
        .destination-cycle span:nth-child(4) { animation-delay: 9s; }

        @keyframes cycleWords {
            0% { opacity: 0; transform: translateY(18px); }
            8% { opacity: 1; transform: translateY(0); }
            25% { opacity: 1; transform: translateY(0); }
            33% { opacity: 0; transform: translateY(-18px); }
            100% { opacity: 0; transform: translateY(-18px); }
        }

        .hero-facts {
            display: grid;
            gap: 0.7rem;
        }

        .fact-card {
            border: 1px solid var(--line);
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.72);
            padding: 1rem 1rem 0.95rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
            backdrop-filter: blur(14px);
        }

        .fact-label {
            display: block;
            font-size: 0.72rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: rgba(30, 38, 47, 0.56);
            margin-bottom: 0.45rem;
        }

        .fact-value {
            font-family: "Cormorant Garamond", serif;
            font-size: 1.45rem;
            line-height: 1.05;
            color: var(--cream);
        }

        .sticky-nav {
            position: sticky;
            top: 0.6rem;
            z-index: 35;
            margin: 0.6rem 0 1.15rem;
            padding: 0.6rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(159, 119, 52, 0.18);
            background: rgba(255, 255, 255, 0.72);
            backdrop-filter: blur(18px);
            display: flex;
            justify-content: space-between;
            gap: 0.6rem;
            flex-wrap: wrap;
            box-shadow: 0 18px 50px rgba(30, 38, 47, 0.1);
        }

        .nav-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(30, 38, 47, 0.08);
            color: var(--muted);
            background: rgba(255,255,255,0.86);
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .nav-pill::before {
            content: "•";
            color: var(--gold);
            font-size: 1rem;
            line-height: 0;
        }

        .stForm {
            border: 1px solid rgba(30, 38, 47, 0.08);
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(248, 242, 232, 0.86));
            box-shadow: var(--shadow);
            padding: 1rem 1rem 0.5rem;
        }

        [data-testid="stForm"] {
            border: 1px solid rgba(30, 38, 47, 0.08);
            border-radius: 28px;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(248, 242, 232, 0.86));
            box-shadow: var(--shadow);
            padding: 1rem 1rem 0.5rem;
        }

        [data-testid="stForm"] label,
        label {
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.72rem !important;
            color: rgba(30, 38, 47, 0.68) !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] > div,
        textarea {
            background: rgba(255, 255, 255, 0.95) !important;
            color: var(--cream) !important;
            border: 1px solid rgba(159, 119, 52, 0.2) !important;
            border-radius: 18px !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
        }

        [data-baseweb="select"] span {
            color: var(--cream) !important;
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stDateInput"] input:focus,
        textarea:focus {
            border-color: rgba(159, 119, 52, 0.55) !important;
            box-shadow: 0 0 0 3px rgba(159, 119, 52, 0.12) !important;
        }

        .stButton > button {
            border: 1px solid rgba(159, 119, 52, 0.3);
            background: linear-gradient(135deg, #9f7734, #d8bd79);
            color: #fffaf2 !important;
            border-radius: 999px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.78rem 1.35rem;
            box-shadow: 0 18px 30px rgba(159, 119, 52, 0.16);
            transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 24px 40px rgba(159, 119, 52, 0.22);
            filter: brightness(1.02);
        }

        .stCheckbox label {
            text-transform: none !important;
            letter-spacing: 0.02em !important;
            font-size: 0.95rem !important;
        }

        [data-testid="stExpander"] {
            border: 1px solid rgba(30, 38, 47, 0.08);
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.72);
            overflow: hidden;
            box-shadow: 0 16px 40px rgba(30, 38, 47, 0.08);
            transition: transform 180ms ease, border-color 180ms ease;
        }

        [data-testid="stExpander"]:hover {
            transform: translateY(-2px);
            border-color: rgba(159, 119, 52, 0.28);
        }

        [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(30, 38, 47, 0.08);
        }

        [data-testid="stDataFrame"] * {
            color: #1e262f !important;
        }

        .section-divider {
            height: 1px;
            margin: 1.2rem 0 1rem;
            background: linear-gradient(90deg, transparent, rgba(159, 119, 52, 0.55), transparent);
        }

        .map-placeholder {
            margin-top: 1.2rem;
            padding: 1.4rem;
            border-radius: 28px;
            border: 1px solid rgba(30, 38, 47, 0.08);
            background:
                linear-gradient(135deg, rgba(255,255,255,0.72), rgba(255,255,255,0.6)),
                radial-gradient(circle at 20% 20%, rgba(159, 119, 52, 0.12), transparent 26%),
                radial-gradient(circle at 82% 30%, rgba(214, 109, 93, 0.1), transparent 22%),
                rgba(255, 255, 255, 0.62);
            box-shadow: var(--shadow);
        }

        .map-grid {
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 1rem;
            align-items: center;
        }

        .map-surface {
            min-height: 220px;
            border-radius: 22px;
            border: 1px solid rgba(30, 38, 47, 0.08);
            background:
                linear-gradient(120deg, rgba(159, 119, 52, 0.16), transparent 30%),
                linear-gradient(45deg, rgba(214, 109, 93, 0.12), transparent 34%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(242, 235, 224, 0.96));
            position: relative;
            overflow: hidden;
        }

        .map-surface::before,
        .map-surface::after {
            content: "";
            position: absolute;
            border-radius: 999px;
            background: rgba(159, 119, 52, 0.18);
        }

        .map-surface::before {
            width: 120px;
            height: 120px;
            right: 18%;
            top: 22%;
        }

        .map-surface::after {
            width: 18px;
            height: 18px;
            left: 22%;
            top: 58%;
            background: rgba(214, 109, 93, 0.9);
            box-shadow: 0 0 0 14px rgba(214, 109, 93, 0.1);
        }

        .footer-note {
            color: rgba(30, 38, 47, 0.64);
            line-height: 1.7;
        }

        .stInfo {
            border-radius: 18px;
            border: 1px solid rgba(159, 119, 52, 0.18);
            background: rgba(255, 255, 255, 0.82);
            color: var(--cream);
        }

        @media (max-width: 900px) {
            .hero-grid,
            .map-grid {
                grid-template-columns: 1fr;
            }

            h1 {
                font-size: clamp(2.5rem, 12vw, 4rem) !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Free Travel Planner", layout="wide")
inject_travel_branding()

st.markdown(
    """
    <div class="travel-hero">
      <div class="hero-grid">
        <div class="hero-copy">
          <div class="hero-kicker">Editorial travel studio</div>
          <h2 style="margin: 0 0 0.35rem 0;">Plan journeys that feel tactile, cinematic, and carefully composed.</h2>
          <p>
            Build an itinerary with the polish of a premium travel magazine and the clarity of a modern planner.
            <span class="destination-cycle">
              <span>Bali</span>
              <span>Tokyo</span>
              <span>Paris</span>
              <span>Lisbon</span>
            </span>
          </p>
        </div>
        <div class="hero-facts">
          <div class="fact-card">
            <span class="fact-label">Style</span>
            <div class="fact-value">Ivory, terracotta, and champagne gold</div>
          </div>
          <div class="fact-card">
            <span class="fact-label">Experience</span>
            <div class="fact-value">Travel planning with atmosphere, motion, and crafted detail</div>
          </div>
        </div>
      </div>
    </div>
    <div class="sticky-nav">
      <span class="nav-pill">Itinerary</span>
      <span class="nav-pill">Hotels</span>
      <span class="nav-pill">Restaurants</span>
      <span class="nav-pill">Attractions</span>
      <span class="nav-pill">Budget</span>
      <span class="nav-pill">Tips</span>
      <span class="nav-pill">Weather</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Free Travel Planner")
st.write("A simple travel planner that uses free public APIs, plus Gemini when your key is available.")

with st.form("trip_form"):
    c1, c2 = st.columns(2)
    with c1:
        destination = st.text_input("Destination", placeholder="e.g. Tokyo, Paris, Bali")
        departure_city = st.text_input("Departure city", placeholder="e.g. Mumbai, New York")
        start_date = st.date_input("Start date", value=date.today() + timedelta(days=30))
    with c2:
        duration = st.number_input("Trip duration (days)", min_value=1, max_value=30, value=5)
        budget = st.selectbox("Budget", ["budget", "mid-range", "luxury"], index=1)
        travelers = st.number_input("Travelers", min_value=1, max_value=10, value=2)

    interests = st.multiselect(
        "Interests",
        ["food", "history", "nature", "nightlife", "shopping", "art", "relaxation", "adventure"],
        default=["food", "history", "nature"],
    )

    use_gemini = st.checkbox("Use Gemini to refine the itinerary", value=gemini_is_available())

    submitted = st.form_submit_button("Build itinerary")

if submitted:
    if not destination.strip():
        st.error("Please enter a destination.")
    else:
        travel_dates = [start_date + timedelta(days=i) for i in range(int(duration))]
        local_plan = build_itinerary(destination.strip(), travel_dates, interests, budget, int(travelers), departure_city.strip())
        gemini_plan = None
        if use_gemini:
            with st.spinner("Gemini is refining the itinerary..."):
                gemini_plan = refine_with_gemini(local_plan, travel_dates, interests, budget, int(travelers))
        plan = normalize_gemini_plan(local_plan, gemini_plan)

        left, right = st.columns([1.2, 0.8])
        with left:
            st.subheader(plan["destination_label"])
            st.write(plan["summary"])
            st.write(f"Departure city: {plan['departure_city'] or 'Not provided'}")
            st.write(f"Visa tip: {plan['visa']}")
            st.write(f"Season tip: {plan['season']}")
            st.subheader("Day-by-day itinerary")
            for idx, day in enumerate(as_list(plan.get("itinerary_days")), start=1):
                if isinstance(day, dict):
                    title = f"Day {day.get('day', idx)} - {str(day.get('focus', 'Overview')).title()}"
                    morning = day.get("morning", "")
                    afternoon = day.get("afternoon", "")
                    evening = day.get("evening", "")
                    hotel_note = day.get("hotel_note", "")
                else:
                    title = f"Day {idx}"
                    morning = afternoon = evening = hotel_note = str(day)
                with st.expander(title):
                    st.write(f"Morning: {morning}")
                    st.write(f"Afternoon: {afternoon}")
                    st.write(f"Evening: {evening}")
                    st.write(f"Stay note: {hotel_note}")

        with right:
            st.subheader("Estimated Budget")
            if isinstance(plan.get("budget"), dict):
                budget_rows = [{"category": key.title(), "estimated_cost": value} for key, value in plan["budget"].items()]
                st.dataframe(budget_rows, use_container_width=True, hide_index=True)
            else:
                st.write(plan.get("budget"))
            st.subheader("Suggested Hotels")
            st.write("\n".join(f"- {item}" for item in render_items(plan["hotels"])))
            st.subheader("Suggested Restaurants")
            st.write("\n".join(f"- {item}" for item in render_items(plan["restaurants"])))
            st.subheader("Suggested Attractions")
            st.write("\n".join(f"- {item}" for item in render_items(plan["attractions"])))
            st.subheader("Travel Tips")
            st.write("\n".join(f"- {tip}" for tip in render_items(plan["tips"])))
            if plan.get("avoid"):
                st.subheader("What to Avoid")
                st.write("\n".join(f"- {item}" for item in render_items(plan["avoid"])))
            if plan.get("alternatives"):
                st.subheader("Budget Flex Options")
                st.write("\n".join(f"- {item}" for item in render_items(plan["alternatives"])))

        if plan["weather"]:
            st.subheader("Weather")
            daily = plan["weather"].get("daily", {})
            if daily:
                weather_rows = []
                days = daily.get("time", [])
                tmax = daily.get("temperature_2m_max", [])
                tmin = daily.get("temperature_2m_min", [])
                rain = daily.get("precipitation_probability_max", [])
                for idx, day in enumerate(days[:5]):
                    weather_rows.append(
                        {
                            "date": day,
                            "high": tmax[idx] if idx < len(tmax) else None,
                            "low": tmin[idx] if idx < len(tmin) else None,
                            "rain %": rain[idx] if idx < len(rain) else None,
                        }
                    )
                st.dataframe(weather_rows, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div class="map-placeholder">
              <div class="map-grid">
                <div>
                  <h3 style="margin-top:0;">Destination map</h3>
                  <p class="footer-note">
                    A full-width destination map placeholder lives here to keep the layout feeling editorial and grounded.
                    The visual treatment is intentionally atmospheric so the trip reads like a premium travel brand.
                  </p>
                </div>
                <div class="map-surface" aria-hidden="true"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    st.info("Fill in your trip details and click Build itinerary.")
    st.subheader("What this app can answer")
    st.write(
        "- A day-by-day itinerary\n"
        "- Hotel, restaurant, and attraction suggestions\n"
        "- Budget estimates\n"
        "- Visa and seasonal travel tips\n"
        "- Tourist-trap and safety reminders"
    )
    st.markdown(
        """
        <div class="map-placeholder">
          <div class="map-grid">
            <div>
              <h3 style="margin-top:0;">Destination map</h3>
              <p class="footer-note">
                A full-width map area will anchor the page once you build a trip, keeping the design consistent
                even before results are generated.
              </p>
            </div>
            <div class="map-surface" aria-hidden="true"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
