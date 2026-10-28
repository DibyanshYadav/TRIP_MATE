# tools/travel_links.py
"""
Builds ready-to-open flight and hotel search links (Google Flights, Skyscanner,
Kayak, Booking.com, Google Hotels, Google Maps).

The links are created by CODE, not by the language model, so they can never be
invented or mistyped. They open live search pages: TripMate does not book anything.
"""
import re
from datetime import datetime
from urllib.parse import quote, quote_plus

try:
    from tools.flight_tools import resolve_location_to_iata
except Exception:          # airportsdata etc. missing: still build the name-based links
    resolve_location_to_iata = None


def _parse_date(text: str):
    """Accepts only YYYY-MM-DD. Returns a date or None."""
    try:
        return datetime.strptime((text or "").strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _iata(place: str):
    if not resolve_location_to_iata or not place:
        return None
    try:
        return resolve_location_to_iata(place)
    except Exception:
        return None


def _clean(text: str) -> str:
    """Remove characters that would break a markdown link label."""
    return re.sub(r"[\[\]()]", "", str(text)).strip()


def build_travel_links(
    origin: str,
    destination: str,
    depart_date: str = "",
    return_date: str = "",
    hotels: str = "",
) -> str:
    origin = _clean(origin)
    destination = _clean(destination)
    if not destination:
        return "Link generation failed: destination is required."

    dep = _parse_date(depart_date)
    ret = _parse_date(return_date)
    if dep and ret and ret < dep:
        ret = None                       # nonsense return date: treat as one-way

    parts = ["### 🔗 Quick links (open on your phone)"]

    # ------------------------------------------------------------- flights
    if origin:
        o_iata, d_iata = _iata(origin), _iata(destination)

        q = f"Flights from {origin} to {destination}"
        if dep:
            q += f" on {dep.isoformat()}"
            if ret:
                q += f" through {ret.isoformat()}"
        google_flights = "https://www.google.com/travel/flights?q=" + quote(q)

        flight_links = [f"- [Google Flights]({google_flights})"]

        if o_iata and d_iata and dep:
            sky = f"https://www.skyscanner.net/transport/flights/{o_iata.lower()}/{d_iata.lower()}/{dep:%y%m%d}/"
            if ret:
                sky += f"{ret:%y%m%d}/"
            kayak = f"https://www.kayak.com/flights/{o_iata}-{d_iata}/{dep.isoformat()}"
            if ret:
                kayak += f"/{ret.isoformat()}"
            flight_links.append(f"- [Skyscanner]({sky})")
            flight_links.append(f"- [Kayak]({kayak})")

        route = f"{o_iata} → {d_iata}" if (o_iata and d_iata) else f"{origin} → {destination}"
        when = f", {dep.isoformat()}" if dep else ""
        if dep and ret:
            when += f" to {ret.isoformat()}"
        parts.append(f"**Flights** ({route}{when})\n" + "\n".join(flight_links))

    # -------------------------------------------------------------- hotels
    booking = f"https://www.booking.com/searchresults.html?ss={quote_plus(destination)}"
    if dep and ret:
        booking += f"&checkin={dep.isoformat()}&checkout={ret.isoformat()}"
    google_hotels = "https://www.google.com/travel/search?q=" + quote(f"hotels in {destination}")

    parts.append(
        f"**Hotels in {destination}**\n"
        f"- [Search on Booking.com]({booking})\n"
        f"- [Search on Google Hotels]({google_hotels})"
    )

    # ------------------------------------- shortlisted hotels (map + directions)
    names = [_clean(h) for h in (hotels or "").split(";")]
    names = [n for n in names if n][:6]
    if names:
        lines = []
        for name in names:
            maps = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(f"{name} {destination}")
            lines.append(f"- [{name}]({maps})")
        parts.append("**Your shortlisted hotels (map, photos, directions)**\n" + "\n".join(lines))

    parts.append(
        "_These links open live search pages, so prices and availability may differ "
        "from what is shown above. TripMate cannot book anything: always confirm on the site._"
    )

    return "\n\n".join(parts)


if __name__ == "__main__":
    print(build_travel_links(
        "Delhi", "London", "2026-12-05", "2026-12-09",
        "The Z Hotel Piccadilly; Premier Inn London County Hall",
    ))