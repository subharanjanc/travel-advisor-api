"""The advisor's brain, lifted out of travel_advisor.ipynb.

Kept separate from index.py so the logic can be tested without starting a server.
Everything here is stateless: no conversation is held in memory, because a
serverless function has nowhere to hold one. Continuity comes from the
previous_response_id the client sends back.
"""

import datetime
import json
import os
import time

from openai import OpenAI

MODEL = "gpt-6-luna"
PRICES = {"in": 0.10, "out": 0.50}          # USD per 1M tokens, estimate only

MAX_TOOL_ROUNDS = 3
MAX_TOOL_CALLS = 6
MAX_MESSAGE_CHARS = 2000


def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set on the server.")
    return OpenAI(api_key=key, timeout=60.0, max_retries=1)


# --------------------------------------------------------------- climate data
# MOCK. Approximate typical conditions, hand-entered. Never a forecast.
MONTH_SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}

MONTH_NAMES = ["january", "february", "march", "april", "may", "june",
               "july", "august", "september", "october", "november", "december"]

SOUTH_EUROPE = {
    "lisbon":    {"summer": "28C, dry and bright, almost no rain",
                  "autumn": "22C, warm days, first rain returns",
                  "winter": "15C, mild but wet, 11 rainy days",
                  "spring": "20C, changeable, pleasant for walking"},
    "porto":     {"summer": "25C, warm, cooler than Lisbon, Atlantic breeze",
                  "autumn": "19C, wet, grey mornings",
                  "winter": "14C, mild but the wettest city on this list",
                  "spring": "18C, showers between bright spells"},
    "madrid":    {"summer": "33C, very hot and dry, fierce midday sun",
                  "autumn": "19C, crisp and clear",
                  "winter": "10C, cold nights, dry, occasional frost",
                  "spring": "20C, warm days and cold evenings"},
    "barcelona": {"summer": "29C, hot and humid, busy beaches",
                  "autumn": "22C, warm sea, heaviest rain of the year",
                  "winter": "14C, mild, mostly dry",
                  "spring": "20C, comfortable, light showers"},
    "seville":   {"summer": "36C, extreme heat, much of the city closes midday",
                  "autumn": "26C, still warm into November",
                  "winter": "16C, mild days, cool nights, some rain",
                  "spring": "25C, the best time to visit"},
    "valencia":  {"summer": "30C, humid, warm sea",
                  "autumn": "24C, warm, sudden heavy downpours",
                  "winter": "16C, mild and mostly dry",
                  "spring": "22C, warm and settled"},
    "rome":      {"summer": "31C, hot and dry, little shade in the forums",
                  "autumn": "22C, warm, rain increases through November",
                  "winter": "13C, damp and grey, short queues",
                  "spring": "21C, warm, busy around Easter"},
    "florence":  {"summer": "32C, hot and close, inland so no sea breeze",
                  "autumn": "20C, mild, wet in November",
                  "winter": "11C, damp and cold, quiet museums",
                  "spring": "20C, mild and bright"},
    "venice":    {"summer": "29C, humid, mosquitoes, very crowded",
                  "autumn": "18C, acqua alta flooding is common",
                  "winter": "7C, cold and foggy, fewest visitors",
                  "spring": "18C, mild, rising crowds"},
    "naples":    {"summer": "30C, hot, hazy, crowded coast",
                  "autumn": "22C, warm, heavy autumn rain",
                  "winter": "14C, mild and wet",
                  "spring": "21C, warm and clear"},
    "athens":    {"summer": "34C, very hot and dry, little shade on the Acropolis",
                  "autumn": "24C, warm, pleasant for ruins",
                  "winter": "14C, cool, some rain, quiet sites",
                  "spring": "22C, warm, wildflowers"},
    "split":     {"summer": "30C, hot, warm sea, peak ferry season",
                  "autumn": "21C, warm sea, more rain",
                  "winter": "11C, wet, many islands shut down",
                  "spring": "19C, mild, sea still cold"},
    "dubrovnik": {"summer": "29C, hot, very crowded old town",
                  "autumn": "21C, warm, wettest of the Croatian coast",
                  "winter": "12C, mild but wet, much is closed",
                  "spring": "19C, pleasant, quiet walls"},
    "palma":     {"summer": "30C, hot and dry, packed",
                  "autumn": "23C, warm sea, occasional storms",
                  "winter": "15C, mild, many resorts closed",
                  "spring": "20C, warm, good for cycling"},
    "nice":      {"summer": "28C, hot, busy promenade",
                  "autumn": "21C, warm, heavy October rain",
                  "winter": "13C, mild and bright for the latitude",
                  "spring": "18C, mild, sea still cold"},
    "valletta":  {"summer": "32C, very hot, strong sun, little shade",
                  "autumn": "25C, warm sea well into October",
                  "winter": "16C, mild, windy, the rainy season",
                  "spring": "22C, warm and dry"},
}

CLIMATE = {(city, month): seasons[MONTH_SEASON[i + 1]]
           for city, seasons in SOUTH_EUROPE.items()
           for i, month in enumerate(MONTH_NAMES)}

CLIMATE.update({
    ("lisbon", "may"):     "21C, mostly sunny, 6 rainy days in the month",
    ("lisbon", "january"): "15C, mild but wet, 11 rainy days",
    ("kyoto", "april"):    "18C, cherry blossom season, occasional showers",
    ("kyoto", "august"):   "33C, very humid, afternoon thunderstorms",
    ("reykjavik", "june"): "12C, daylight almost round the clock, windy",
})

COVERED_CITIES = sorted({city for city, _ in CLIMATE})


def get_weather(city: str, month: str) -> str:
    return CLIMATE.get((city.strip().lower(), month.strip().lower()),
                       "No climate data on file for that city and month")


# --------------------------------------------------------------- the prompt
ITINERARY_SPEC = """When, and only when, you have destination, duration and interests, produce
the itinerary in exactly this Markdown structure and nothing else:

# <trip title>

<two or three sentence summary>

## Day-by-day
### Day 1 - <theme>
- morning: ...
- afternoon: ...
- evening: ...
(repeat for every day of the trip)

## Practical tips
- <4 to 6 bullets: getting around, booking, timing, etiquette>

## Estimated budget
| item | per person |
|---|---|
| accommodation | EUR ... |
| food | EUR ... |
| transport | EUR ... |
| activities | EUR ... |
| **total** | **EUR ...** |
"""


def build_system() -> str:
    """Built per request so the date is today's, not the date of the last deploy."""
    today = datetime.date.today()
    next_month = (today.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
    may_year = today.year + (0 if today.month < 5 else 1)
    return f"""You are a travel advisor. You plan realistic trips through conversation.

PHASE 1 - interview.
Collect, in as few questions as possible:
  1. destination
  2. trip duration
  3. interests (food, museums, hiking, beaches, nightlife, shopping, ...)
Budget, travel month, pace and constraints are optional - ask for them at most once, together,
and never block the itinerary on them.

Rules for this phase:
- Ask about things you do not yet know. Never re-ask something already answered.
- Ask for at most two information fields in one message.
- If the user contradicts themselves, say what the conflict is and ask which one is right.
  Do not guess, and do not continue planning until it is resolved.
- If the user says they do not mind or asks you to decide, treat that as an answer and move on.

PHASE 2 - the itinerary.
As soon as you have destination, duration and interests, stop asking questions and write the
plan. Do not announce it. Just write it.

{ITINERARY_SPEC}

TODAY'S DATE is {today:%A %d %B %Y}.
- A month named without a year means its next occurrence. "May" means {may_year}.
- "Next month" means {next_month:%B %Y}.
- Never plan a trip for a date that has already passed. Say so and ask which year they mean.
- Knowing the date tells you nothing about prices, opening hours, availability or weather.

Always:
- Keep the plan physically possible: no three cities in a day.
- Budgets are rough estimates in EUR. Say so once, in the tips.
- You may call get_weather. It returns MOCK typical climate, never a live forecast. Label it as
  mock climate data when you use it. If you have not called it, do not state the weather.
- Never invent opening hours, prices of specific venues, or availability.
- You only discuss travel planning. Decline anything else briefly and return to the trip.
"""


TOOLS = [{
    "type": "function",
    "name": "get_weather",
    "strict": True,
    "description": ("MOCK typical climate for a city/month, not a forecast. Coverage is southern "
                    "Europe plus a few others; returns 'No climate data on file' for anywhere "
                    "else. Use before mentioning weather, packing or outdoor plans."),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. Lisbon"},
            "month": {"type": "string", "description": "Month name, e.g. May"},
        },
        "required": ["city", "month"],
        "additionalProperties": False,
    },
}]


def run_tool(item) -> str:
    """Allowlist, then validate the model's arguments before touching Python."""
    if item.name != "get_weather":
        return json.dumps({"error": "Unsupported tool."})
    try:
        args = json.loads(item.arguments)
        if not isinstance(args, dict) or set(args) != {"city", "month"}:
            raise ValueError("Expected city and month only.")
        if any(not isinstance(v, str) or not v.strip() or len(v) > 100 for v in args.values()):
            raise ValueError("City and month must be short non-empty strings.")
        if args["month"].strip().lower() not in MONTH_NAMES:
            raise ValueError("Use a full English month name.")
        return json.dumps({"source": "mock climate table; not a live forecast",
                           "result": get_weather(**args)})
    except (ValueError, TypeError) as error:
        return json.dumps({"error": str(error)})


def valid_response_id(value) -> bool:
    """A bad id makes the upstream API throw. Reject it here with a clear message instead.

    Found while testing the class demo backend: any malformed id there returns a bare 502.
    """
    if value in (None, ""):
        return True
    return (isinstance(value, str) and value.startswith("resp_")
            and len(value) <= 200 and value[5:].isalnum())


def answer(message: str, previous_response_id: str | None = None) -> dict:
    """One turn. Returns the reply, the id to send next time, and what it cost."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must not be empty.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError(f"Message must be at most {MAX_MESSAGE_CHARS} characters.")
    if not valid_response_id(previous_response_id):
        raise ValueError("previous_response_id must be empty or look like 'resp_...'.")

    client = get_client()
    started = time.perf_counter()
    tokens_in = tokens_out = 0
    tools_used = []

    request = {"model": MODEL, "input": message, "tools": TOOLS,
               "instructions": build_system(), "max_output_tokens": 4000}
    if previous_response_id:
        request["previous_response_id"] = previous_response_id

    response = client.responses.create(**request)
    tokens_in += response.usage.input_tokens
    tokens_out += response.usage.output_tokens

    rounds = calls_used = 0
    while True:
        calls = [i for i in response.output if i.type == "function_call"]
        if not calls:
            break
        if rounds >= MAX_TOOL_ROUNDS or calls_used + len(calls) > MAX_TOOL_CALLS:
            raise RuntimeError("Tool-call limit reached. Please simplify the request.")
        outputs = []
        for item in calls:
            result = run_tool(item)
            tools_used.append(item.name)
            outputs.append({"type": "function_call_output",
                            "call_id": item.call_id, "output": result})
        rounds += 1
        calls_used += len(calls)
        response = client.responses.create(
            model=MODEL, input=outputs, tools=TOOLS,
            previous_response_id=response.id, max_output_tokens=4000)
        tokens_in += response.usage.input_tokens
        tokens_out += response.usage.output_tokens

    reply = (response.output_text or "").strip()
    if not reply:
        raise RuntimeError("The model returned no text. Please retry.")

    cost = (tokens_in * PRICES["in"] + tokens_out * PRICES["out"]) / 1e6
    return {
        "answer": reply,
        "response_id": response.id,
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out,
                  "tools_called": tools_used, "seconds": round(time.perf_counter() - started, 2)},
        "estimated_cost_usd": round(cost, 6),
    }
