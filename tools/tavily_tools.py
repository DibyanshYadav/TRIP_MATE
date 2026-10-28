# tools/tavily_tools.py
import os
import re
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from tavily import TavilyClient

load_dotenv()

client = TavilyClient(
    api_key=os.getenv('TAVILY_API_KEY')
)


# =====================================================================
# 1) General search (unchanged): used by weather / transport tools
# =====================================================================
def tavily_search(query):
    response = client.search(query=query, max_results=5)

    results = []

    for i, r in enumerate(response["results"], 1):
        title = r.get("title", "Unknown")
        url = r.get("url", "")
        snippet = r.get("content", "").strip()
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    return "\n\n".join(results)


# =====================================================================
# 2) Hotel search: returns REAL hotel names
#    Tavily -> short text -> small LLM extracts hotels -> code verifies every
#    name really appears in the text (LLM cannot invent hotels) -> table
# =====================================================================
_extract_llm = ChatGroq(
    model=os.getenv("EXTRACT_MODEL", "llama-3.1-8b-instant"),
    temperature=0,
    max_tokens=900,
)


class Hotel(BaseModel):
    name: str = Field(description="Exact hotel name as written in the text (a real property, not a list title)")
    area: Optional[str] = Field(None, description="Neighbourhood if stated")
    price_per_night: Optional[float] = Field(None, description="Nightly price number if stated")
    currency: Optional[str] = Field(None, description="Currency code of that price, e.g. AED, USD, INR")
    near_metro: Optional[str] = Field(None, description="Nearest metro station if stated")
    url: Optional[str] = Field(None, description="URL of the source block it came from")


class HotelList(BaseModel):
    hotels: List[Hotel]


_JUNK = re.compile(r"^(best|top|cheap|cheapest|budget|\d+)\b|\bhotels? (in|near)\b|\bguide\b", re.I)


def _hotel_queries(dest, checkin, budget, currency, near_metro, area):
    month = (checkin or "")[:7]
    price = f" under {int(budget)} {currency or ''}".rstrip() if budget else ""
    qs = [
        f"best budget hotels in {dest}{price} per night",
        f"cheap 3 star hotels {dest} price per night {month}".strip(),
    ]
    if area:
        qs.append(f"budget hotels in {area} {dest}")
    if near_metro:
        qs.append(f"budget hotels near metro station {dest}")
    return qs


def _search_blob(queries, per_query=5, max_chars=400):
    seen, blocks, urls = set(), [], set()
    for q in queries:
        try:
            res = client.search(query=q, max_results=per_query)
        except Exception as e:
            print(f"[tavily_tools] search error for '{q}': {e}")
            continue
        for r in res.get("results", []):
            url = r.get("url", "")
            if not url or url in seen:
                continue
            seen.add(url)
            urls.add(url)
            blocks.append(f"TITLE: {r.get('title','')}\nURL: {url}\nTEXT: {r.get('content','')[:max_chars]}\n")
    return "\n".join(blocks), urls


def search_hotels(
    destination: str,
    checkin: str = "",
    checkout: str = "",
    max_price_per_night: float = 0,
    currency: str = "",
    near_metro: bool = False,
    area: str = "",
) -> str:
    blob, urls = _search_blob(
        _hotel_queries(destination, checkin, max_price_per_night, currency, near_metro, area)
    )
    if not blob.strip():
        return "HOTEL SEARCH FAILED: no results from the search API. Do not invent hotel names."

    prompt = (
        f"Extract real hotels in or near {destination} from the search text.\n"
        "Rules:\n"
        "- Only hotels explicitly named in the text. NEVER invent names, prices or ratings.\n"
        "- Skip article titles like 'Best hotels in X'. Only actual hotel properties.\n"
        "- Use null for anything not stated. url = the URL of the block the hotel appears in.\n"
        "- At most 8 hotels.\n\n"
        f"SEARCH TEXT:\n{blob}"
    )
    try:
        result: HotelList = _extract_llm.with_structured_output(HotelList).invoke(prompt)
    except Exception as e:
        print(f"[tavily_tools] hotel extraction failed: {e}")
        return "HOTEL SEARCH FAILED: could not read the results. Do not invent hotel names."

    blob_l = blob.lower()
    hotels, names = [], set()
    for h in result.hotels:
        name = h.name.strip()
        key = name.lower()
        if len(name) < 4 or key in names or _JUNK.search(name):
            continue
        if key not in blob_l:              # anti-hallucination: name must appear in the text
            continue
        if h.url and h.url not in urls:
            h.url = None
        names.add(key)
        hotels.append(h)

    hotels.sort(key=lambda h: h.price_per_night is None)   # priced hotels first
    hotels = hotels[:5]

    if not hotels:
        return (
            f"NO VERIFIED HOTEL NAMES FOUND for {destination}. Tell the user honestly, "
            "suggest raising the budget or another area, and rely on the booking links. "
            "Do not invent hotel names."
        )

    rows = ["| # | Hotel | Area | Price / night | Near metro | Source |", "|---|---|---|---|---|---|"]
    for i, h in enumerate(hotels, 1):
        price = f"{h.currency or ''} {h.price_per_night:,.0f}".strip() if h.price_per_night else "check site"
        src = f"[open]({h.url})" if h.url else "-"
        rows.append(f"| {i} | **{h.name}** | {h.area or '-'} | {price} | {h.near_metro or '-'} | {src} |")

    return (
        "\n".join(rows)
        + "\n\nPrices come from web snippets and may be outdated: confirm on the booking site."
        + "\nHOTEL_NAMES: " + "; ".join(h.name for h in hotels)
    )