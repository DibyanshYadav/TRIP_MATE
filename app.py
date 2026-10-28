from pathlib import Path
import json
import os
import re
import threading
import time
import traceback
import uuid
from datetime import datetime

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# `app` in backend.py is the compiled LangGraph, so alias it to avoid clashing with FastAPI's `app`
# (importing backend also runs load_dotenv(), so the os.getenv calls below see your .env)
from backend import app as travel_graph, itineary_agent
from tools.travel_links import build_travel_links


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TripMate",
    description="MultiAgent Trip Planner with FastAPI frontend",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ---------------------------------------------------------------- map pins
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim policy: use a real, identifying User-Agent.
# Put GEO_CONTACT_EMAIL=you@example.com in your .env
GEO_CONTACT_EMAIL = os.getenv("GEO_CONTACT_EMAIL", "your@email.com")
GEO_USER_AGENT = f"TripMateAI/1.0 (contact: {GEO_CONTACT_EMAIL})"
MAX_PINS = 10                      # each uncached lookup costs ~1s (Nominatim rate limit)

GEO_CACHE_FILE = BASE_DIR / "geo_cache.json"
_geo_lock = threading.Lock()
_last_geo_call = 0.0


def _load_geo_cache() -> dict:
    try:
        with open(GEO_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_geo_cache():
    """Write the cache atomically so a crash never leaves a half-written file."""
    try:
        tmp = GEO_CACHE_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_geo_cache, f, ensure_ascii=False)
        os.replace(tmp, GEO_CACHE_FILE)
    except Exception as e:
        print("Could not save geo cache:", e)


_geo_cache: dict = _load_geo_cache()    # key -> [lat, lon] or None (place not found)


def geocode(place: str):
    """Place name -> (lat, lon) using free OpenStreetMap Nominatim.
    Cached on disk and rate limited. If the full name is not found it retries
    with a shorter query ("Name, Country")."""
    global _last_geo_call
    place = place.replace("\u2019", "'").replace("\u2018", "'").strip()
    key = place.lower()

    parts = [p.strip() for p in place.split(",") if p.strip()]
    candidates = [place]
    if len(parts) >= 3:
        candidates.append(parts[0] + ", " + parts[-1])

    with _geo_lock:
        if key in _geo_cache:
            cached = _geo_cache[key]
            return tuple(cached) if cached else None

        loc = None
        for q in candidates:
            wait = 1.1 - (time.time() - _last_geo_call)   # max ~1 request per second
            if wait > 0:
                time.sleep(wait)
            try:
                r = requests.get(
                    NOMINATIM_URL,
                    params={"q": q, "format": "json", "limit": 1},
                    headers={"User-Agent": GEO_USER_AGENT},
                    timeout=10,
                )
                _last_geo_call = time.time()
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print("Geocode failed:", q, e)
                return None          # network error: do not cache, try again next time
            if data:
                loc = [float(data[0]["lat"]), float(data[0]["lon"])]
                break

        _geo_cache[key] = loc
        _save_geo_cache()
        return tuple(loc) if loc else None


def extract_places(text: str):
    """Split the `PLACES_JSON: [...]` line off the text.
    Returns (clean_text, list_of_places)."""
    marker = "PLACES_JSON:"
    idx = text.find(marker)
    if idx == -1:
        return text, []

    clean = text[:idx].rstrip()
    clean = re.sub(r"`+(json)?\s*$", "", clean).rstrip()   # leftover code fence

    rest = text[idx + len(marker):]
    start = rest.find("[")
    if start == -1:
        return clean, []

    try:
        places, _ = json.JSONDecoder().raw_decode(rest[start:])
    except Exception:
        return clean, []

    return clean, places if isinstance(places, list) else []


def build_pins(places: list) -> list:
    pins, seen = [], set()
    for p in places:
        if len(pins) >= MAX_PINS:
            break
        if not isinstance(p, dict):
            continue
        name = str(p.get("name", "")).strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())

        try:
            day = int(p.get("day", 1))
        except (TypeError, ValueError):
            day = 1

        loc = geocode(name)
        if loc:
            pins.append({"name": name, "day": day, "lat": loc[0], "lon": loc[1]})
    return pins


# ---------------------------------------------------------------- helpers
THREAD_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

TOOL_LABELS = {
    "flight_agent": "Searching flights",
    "hotel_agent": "Finding hotels",
    "weather_agent": "Checking the weather",
    "train_local_transport_agent": "Looking up trains and local transport",
    "budget_calculator_agent": "Calculating your budget",
    "itineary_agent": "Building your itinerary",   # spelling matches your backend tool name
    "final_agent": "Writing the final plan",
    "currency_converter_agent": "Converting currencies",
    "travel_links_agent": "Preparing flight and hotel links",
}


def clean_thread_id(thread_id: str | None) -> str:
    """The client picks the thread id, so never trust it blindly."""
    if thread_id and THREAD_RE.match(thread_id):
        return thread_id
    return str(uuid.uuid4())


def sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _turn(msgs: list) -> list:
    """Messages of the current turn (after the latest user message)."""
    last_human = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=0)
    return msgs[last_human:]


def _tool_out(turn: list, name: str) -> str:
    outs = [m.content for m in turn if isinstance(m, ToolMessage) and m.name == name]
    return outs[-1] if outs else ""


def _is_planning_turn(turn: list) -> bool:
    return bool(_tool_out(turn, "hotel_agent") or _tool_out(turn, "flight_agent"))


def _trip_days(facts: dict) -> int:
    nights = facts.get("nights")
    if not nights:
        try:
            d1 = datetime.strptime(facts.get("depart_date", ""), "%Y-%m-%d")
            d2 = datetime.strptime(facts.get("return_date", ""), "%Y-%m-%d")
            nights = (d2 - d1).days
        except Exception:
            nights = 0
    return nights + 1 if nights and nights > 0 else 3


def _hotel_names(hotel_text: str) -> str:
    m = re.search(r"HOTEL_NAMES:\s*(.+)", hotel_text or "")
    return m.group(1).strip() if m else ""


def fallback_links(turn: list, facts: dict) -> str:
    """Build the booking links in code when the model forgot to call travel_links_agent."""
    if not facts.get("destination") or not _is_planning_turn(turn):
        return ""
    try:
        return build_travel_links(
            facts.get("origin") or "",
            facts["destination"],
            facts.get("depart_date") or "",
            facts.get("return_date") or "",
            _hotel_names(_tool_out(turn, "hotel_agent")),
        )
    except Exception as e:
        print("[fallback] links failed:", e)
        return ""


def fallback_itinerary(msgs: list, facts: dict) -> str:
    """Generate the itinerary (with PLACES_JSON) in code when the model forgot to call itineary_agent."""
    turn = _turn(msgs)
    if _tool_out(turn, "itineary_agent") or not facts.get("destination") or not _is_planning_turn(turn):
        return ""
    prefs = " ".join(facts.get("preferences") or []).lower()
    try:
        return itineary_agent.invoke({
            "destination": facts["destination"],
            "num_days": _trip_days(facts),
            "needs_metro": "metro" in prefs,
        })
    except Exception as e:
        print("[fallback] itinerary failed:", e)
        return ""


def build_result(msgs: list, thread_id: str, facts: dict | None = None, use_fallback: bool = True):
    """Turn the final message list into the API result (without map pins).
    Returns (result_dict, raw_places_list)."""
    facts = facts or {}
    turn = _turn(msgs)

    answer = msgs[-1].content
    if not isinstance(answer, str):
        answer = str(answer)

    # map places come from the itinerary tool's PLACES_JSON line
    itinerary_raw = _tool_out(turn, "itineary_agent")
    if not itinerary_raw and use_fallback:
        itinerary_raw = fallback_itinerary(msgs, facts)
    itinerary_clean, places = extract_places(itinerary_raw)

    # never show the raw PLACES_JSON / HOTEL_NAMES helper lines to the user
    answer, _ = extract_places(answer)
    answer = re.sub(r"^\s*HOTEL_NAMES:.*$", "", answer, flags=re.M).rstrip()

    # booking links are built by code (tools/travel_links.py), never by the model
    links_md = _tool_out(turn, "travel_links_agent") or fallback_links(turn, facts)
    if links_md.startswith("### ") and "Quick links" not in answer:
        answer = answer.rstrip() + "\n\n---\n\n" + links_md

    result = {
        "thread_id": thread_id,
        "answer": answer,
        "flight_results": _tool_out(turn, "flight_agent"),
        "hotel_results": _tool_out(turn, "hotel_agent"),
        "itineary": itinerary_clean,
    }
    return result, places


# ---------------------------------------------------------------- api
class TravelRequest(BaseModel):
    messages: str = Field(max_length=4000)
    thread_id: str | None = None


def run_travel_agent(user_input: str, thread_id: str | None = None) -> dict:
    """Blocking version (used by /api/travel). Returns the full result incl. pins."""
    thread_id = clean_thread_id(thread_id)
    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    data, places = build_result(result["messages"], thread_id, result.get("trip_facts"))
    data["pins"] = build_pins(places)
    return data


def stream_travel_agent(user_input: str, thread_id: str):
    """Generator that yields SSE events while the graph runs:
       step    -> a tool started / finished (for the live progress list)
       answer  -> the final text (sent BEFORE geocoding so the user sees it right away)
       pins    -> map pins, sent a bit later because geocoding is slow
       done    -> end of stream
       error   -> something failed
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        for chunk in travel_graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue

                # the model decided to call tools -> they are now running
                if node == "chat":
                    for m in update.get("messages", []):
                        if isinstance(m, AIMessage):
                            for call in (m.tool_calls or []):
                                name = call.get("name", "")
                                yield sse("step", {
                                    "name": name,
                                    "label": TOOL_LABELS.get(name, name),
                                    "state": "running",
                                })

                # tools returned -> mark them finished
                elif node == "tools":
                    for m in update.get("messages", []):
                        if isinstance(m, ToolMessage):
                            name = m.name or ""
                            yield sse("step", {
                                "name": name,
                                "label": TOOL_LABELS.get(name, name),
                                "state": "done",
                            })

        state = travel_graph.get_state(config)
        msgs = state.values.get("messages", [])
        facts = state.values.get("trip_facts") or {}
        result, places = build_result(msgs, thread_id, facts, use_fallback=False)
        yield sse("answer", result)       # user sees the text right away

        if not places:                    # model skipped itineary_agent: build the map places in code
            extra = fallback_itinerary(msgs, facts)
            _, places = extract_places(extra)
            result["itineary"] = extract_places(extra)[0]

        pins = build_pins(places) if places else []
        print(f"[map] places listed by itinerary: {len(places)}, geocoded pins: {len(pins)}")
        yield sse("pins", {
            "pins": pins,
            "places_found": len(places),
            "itinerary_built": bool(result.get("itineary")),
        })
        yield sse("done", {})

    except Exception as e:
        print("STREAM ERROR:", e)
        traceback.print_exc()
        yield sse("error", {"error": str(e)})


@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    """Non-streaming endpoint (kept for testing / simple clients)."""
    try:
        user_message = request_data.messages.strip()
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Message cannot be empty."},
            )

        # graph.invoke is blocking, so run it in a worker thread
        result = await run_in_threadpool(
            run_travel_agent, user_message, request_data.thread_id
        )

        return JSONResponse(content={"success": True, **result})

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/travel/stream")
async def travel_planner_stream(request_data: TravelRequest):
    """Streaming endpoint used by the frontend."""
    user_message = request_data.messages.strip()
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Message cannot be empty."},
        )

    thread_id = clean_thread_id(request_data.thread_id)

    # StreamingResponse runs a normal (sync) generator in a worker thread,
    # so the blocking graph.stream() does not freeze the server.
    return StreamingResponse(
        stream_travel_agent(user_message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",     # stops nginx from buffering the stream
        },
    )


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "AI Travel Planner is Running"}


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)