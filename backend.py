# backend.py  -  TripMate AI (LangGraph)
# app.py does:  from backend import app as travel_graph   -> `app` MUST be the compiled graph.
import json
import os
from datetime import date
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import (
    AIMessage, HumanMessage, RemoveMessage, SystemMessage, trim_messages,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools.flight_tools import search_flights
from tools.tavily_tools import tavily_search, search_hotels
from tools.currency_tool import convert_currency
from tools.travel_links import build_travel_links

load_dotenv()

MAIN_MODEL = os.getenv("MAIN_MODEL", "openai/gpt-oss-120b")
SMALL_MODEL = os.getenv("SMALL_MODEL", "llama-3.1-8b-instant")

llm = ChatGroq(model=MAIN_MODEL, temperature=0.3, max_tokens=3000)       # chat + itinerary
small_llm = ChatGroq(model=SMALL_MODEL, temperature=0, max_tokens=600)    # facts + summary


# =============================================================== trip facts
class TripFacts(BaseModel):
    origin: Optional[str] = Field(None, description="Departure city/country")
    destination: Optional[str] = None
    depart_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    return_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    nights: Optional[int] = None
    travelers: Optional[int] = None
    total_budget: Optional[float] = None
    hotel_budget_per_night: Optional[float] = None
    budget_currency: Optional[str] = Field(None, description="Currency code of the budgets, e.g. INR, AED")
    preferences: Optional[List[str]] = Field(None, description="e.g. '3AC trains', 'metro', 'budget hotels', 'vegetarian'")


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    trip_facts: dict
    summary: str


# ===================================================================== tools
@tool
def flight_agent(origin: str, destination: str) -> str:
    """Live flight status data for a route (NOT fares). Use origin and destination city/country."""
    out = search_flights(f"flights from {origin} to {destination}", limit=5)
    return out[:3500] + (
        "\n\nNOTE: this is live flight status data, not ticket prices. "
        "Give fare estimates only as rough ranges and point the user to the booking links."
    )


@tool
def hotel_agent(
    destination: str,
    checkin: str = "",
    checkout: str = "",
    max_price_per_night: float = 0,
    currency: str = "",
    near_metro: bool = False,
    area: str = "",
) -> str:
    """Find REAL hotel names with prices. Returns a markdown table and a HOTEL_NAMES line.
    Dates are YYYY-MM-DD. max_price_per_night 0 = no limit. currency e.g. AED/INR/USD."""
    return search_hotels(destination, checkin, checkout, max_price_per_night, currency, near_metro, area)


@tool
def weather_agent(destination: str, month: str) -> str:
    """Typical weather for a destination in a given month (e.g. 'December')."""
    return tavily_search(f"{destination} weather in {month} average temperature rainfall")


@tool
def train_local_transport_agent(origin: str, destination: str, preference: str = "") -> str:
    """Trains between cities and local transport (metro, bus, taxi) at the destination.
    preference examples: '3AC train', 'metro'."""
    q = f"{destination} metro bus local transport pass tickets cost {preference}".strip()
    out = tavily_search(q)
    if origin and preference and "train" in preference.lower():
        out += "\n\n---\n\n" + tavily_search(f"train from {origin} to {destination} {preference} fare")
    return out


@tool
def budget_calculator_agent(
    flights: float = 0,
    hotel_per_night: float = 0,
    nights: int = 0,
    food_per_day: float = 0,
    local_transport_per_day: float = 0,
    activities: float = 0,
    misc: float = 0,
    currency: str = "INR",
) -> str:
    """Add up trip costs. All amounts must be in the SAME currency."""
    days = nights + 1
    hotel = hotel_per_night * nights
    food = food_per_day * days
    transport = local_transport_per_day * days
    total = flights + hotel + food + transport + activities + misc
    rows = [
        ("Flights", flights), (f"Hotel ({nights} nights)", hotel), (f"Food ({days} days)", food),
        (f"Local transport ({days} days)", transport), ("Sightseeing", activities), ("Misc", misc),
    ]
    lines = [f"| Item | {currency} |", "|---|---|"]
    lines += [f"| {k} | {v:,.0f} |" for k, v in rows]
    lines.append(f"| **Total** | **{total:,.0f}** |")
    return "\n".join(lines)


@tool
def currency_converter_agent(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert money between currencies using live rates."""
    try:
        return convert_currency(amount, from_currency, to_currency)
    except Exception as e:
        return f"Currency conversion failed: {e}"


@tool
def itineary_agent(
    destination: str,
    num_days: int,
    hotel_area: str = "",
    interests: str = "",
    needs_metro: bool = False,
) -> str:
    """Build a day-by-day itinerary with EXACTLY num_days days. Also returns the PLACES_JSON line for the map."""
    prompt = (
        f"Write a day-by-day sightseeing itinerary for {destination}: EXACTLY {num_days} days "
        f"(Day 1 to Day {num_days}, no extra days). Hotel area: {hotel_area or 'central'}. "
        f"Interests: {interests or 'general sightseeing'}. "
        f"{'Prefer places reachable by metro and name the station.' if needs_metro else ''}\n"
        "Per day: morning / afternoon / evening, one line each, real places only.\n"
        "After the itinerary add ONE final line, nothing after it, in exactly this form:\n"
        'PLACES_JSON: [{"name": "Place, City, Country", "day": 1}, ...]\n'
        "List 6-10 real, map-searchable places (not hotels), each name including city and country."
    )
    return llm.invoke(prompt).content


@tool
def travel_links_agent(
    origin: str, destination: str, depart_date: str = "", return_date: str = "", hotels: str = "",
) -> str:
    """Real booking/search links built by code. depart_date/return_date are YYYY-MM-DD.
    hotels = hotel names separated by ';' (copy from the HOTEL_NAMES line)."""
    return build_travel_links(origin, destination, depart_date, return_date, hotels)


TOOLS = [
    flight_agent, hotel_agent, weather_agent, train_local_transport_agent,
    budget_calculator_agent, currency_converter_agent, itineary_agent, travel_links_agent,
]
llm_with_tools = llm.bind_tools(TOOLS)


# ==================================================================== prompts
SYSTEM_PROMPT = """You are TripMate, a trip-planning assistant. You cannot book anything.

WORKFLOW
- If origin, destination or dates are missing, ask ONE short question and stop. Otherwise plan right away.
- Call tools; never answer flights/hotels/weather/transport from memory.
- hotel_agent: pass the known dates, hotel budget and currency, and near_metro=true if the user wants metro.
- NEVER invent hotel names. Copy the hotel table from hotel_agent EXACTLY. If it found none, say so honestly.
- For every full plan ALWAYS call itineary_agent AND travel_links_agent.
- Call travel_links_agent last, with the hotel names from the HOTEL_NAMES line joined by ';'.
- Do not call the same tool twice. Do not write a 'Quick links' section: the app adds it.
- Honor user preferences. If one cannot apply (e.g. trains between countries), say so in one line.
- Use the KNOWN TRIP FACTS below as the truth for dates, nights and budgets. Do not change them.
- 'N days' means N days; nights = N-1 unless the user gives dates. Keep itinerary day count equal to the trip length.
- Never print the PLACES_JSON line.

FINAL ANSWER FORMAT (markdown, concise):
1. Trip Summary  2. Flights  3. Hotels (the table)  4. Local Transport
5. Itinerary  6. Budget (table)  7. Weather  8. Packing  9. Tips
Flights are status data, not fares: give rough fare ranges and say to verify."""


def _system_text(state: State) -> str:
    text = SYSTEM_PROMPT + f"\n\nToday: {date.today().isoformat()}"
    text += "\nKNOWN TRIP FACTS: " + json.dumps(state.get("trip_facts") or {}, ensure_ascii=False)
    if state.get("summary"):
        text += "\nEARLIER CONVERSATION SUMMARY: " + state["summary"]
    return text


# ===================================================================== nodes
def prepare_node(state: State) -> dict:
    """1) keep trip facts up to date  2) summarize very old messages (saves tokens)."""
    msgs = state["messages"]
    updates: dict = {}

    last_human = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
    if last_human:
        old = state.get("trip_facts") or {}
        try:
            ex = small_llm.with_structured_output(TripFacts).invoke(
                f"Today is {date.today().isoformat()}. Extract trip facts from the user message. "
                "Convert dates like '2nd dec' to YYYY-MM-DD (next occurrence). "
                "Only set fields the user actually stated; leave the rest null.\n"
                f"Known facts: {json.dumps(old)}\nUser message: {last_human.content}"
            )
            new = {k: v for k, v in ex.model_dump().items() if v not in (None, "", [])}
            prefs = sorted(set(old.get("preferences", [])) | set(new.pop("preferences", []) or []))
            merged = {**old, **new}
            if prefs:
                merged["preferences"] = prefs
            updates["trip_facts"] = merged
        except Exception as e:
            print("[prepare] fact extraction failed:", e)

    human_idx = [i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)]
    if len(msgs) > 40 and len(human_idx) >= 3:
        cut = human_idx[-2]                  # keep the last two turns whole (tool pairs stay intact)
        old_msgs = msgs[:cut]
        text = "\n".join(
            f"{m.type}: {str(m.content)[:300]}" for m in old_msgs if m.type in ("human", "ai") and m.content
        )
        try:
            summary = small_llm.invoke(
                "Summarize this trip-planning conversation in under 120 words. Keep decisions, "
                f"dates, budgets, chosen hotels.\nPrevious summary: {state.get('summary','')}\n\n{text}"
            ).content
            updates["summary"] = summary
            updates["messages"] = [RemoveMessage(id=m.id) for m in old_msgs if m.id]
        except Exception as e:
            print("[prepare] summarize failed:", e)

    return updates


def chat_node(state: State) -> dict:
    window = trim_messages(
        state["messages"], strategy="last", max_tokens=30, token_counter=len,
        start_on="human", include_system=False,
    )
    payload = [SystemMessage(content=_system_text(state))] + window
    try:
        reply = llm_with_tools.invoke(payload)
    except Exception as e:                    # Groq sometimes rejects a malformed tool call: retry once
        print("[chat] retrying after error:", e)
        reply = llm_with_tools.invoke(payload)
    return {"messages": [reply]}


# ===================================================================== graph
def make_checkpointer():
    url = os.getenv("DATABASE_URL")           # e.g. postgresql://user:pass@localhost:5432/tripmate
    if url:
        try:
            from psycopg_pool import ConnectionPool
            from langgraph.checkpoint.postgres import PostgresSaver

            pool = ConnectionPool(
                conninfo=url, max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            saver = PostgresSaver(pool)
            saver.setup()
            print("[backend] using Postgres checkpointer")
            return saver
        except Exception as e:
            print("[backend] Postgres unavailable, falling back to memory:", e)
    return MemorySaver()


graph = StateGraph(State)
graph.add_node("prepare", prepare_node)
graph.add_node("chat", chat_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "prepare")
graph.add_edge("prepare", "chat")
graph.add_conditional_edges("chat", tools_condition)   # -> "tools" or END
graph.add_edge("tools", "chat")

app = graph.compile(checkpointer=make_checkpointer())   # <- app.py imports THIS