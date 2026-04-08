# Free Travel Planner

This is a simple interactive travel planner that uses free public APIs and local itinerary logic.

## What it does

- Builds a day-by-day itinerary
- Suggests hotels, restaurants, and attractions
- Estimates travel costs
- Flags visa and seasonal tips
- Warns about common tourist traps

## Free APIs Used

- OpenStreetMap Nominatim for geocoding
- OpenStreetMap Overpass for hotels, restaurants, and attractions
- Open-Meteo for weather
- Wikipedia search for attraction ideas when available
- Gemini API for itinerary refinement when `GEMINI_API_KEY` is set

## Run

```bash
cd "/Users/chaku/Desktop/free_travel_planner"
python3 -m pip install -r requirements.txt
export GEMINI_API_KEY="your_key_here"
streamlit run app.py
```

## Notes

- No Anthropic API is used.
- Gemini is optional, and the app falls back to the local planner when no key is set.
- The app works best with a clear destination, dates, budget, and interests.
- If a free API is rate-limited or temporarily unavailable, the app falls back to local suggestions.
