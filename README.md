# AI Travel Advisor — backend API

Mini-project, Ironhack AI Engineering FT Sept 26.

Plans a trip through conversation: it interviews the traveller for destination,
duration and interests, then writes a day-by-day itinerary with practical tips
and an estimated budget.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/chat` | One turn of the conversation |
| `GET` | `/` | A chat page, so the API is usable without Postman |
| `GET` | `/destinations` | The cities that have climate data |
| `GET` | `/health` | Liveness, model name, coverage |

### `POST /chat`

```json
{
  "message": "I want 4 days in Seville in August. I like food.",
  "previous_response_id": ""
}
```

```json
{
  "answer": "Seville is a great city for eating your way around. **August is extremely hot...",
  "response_id": "resp_...",
  "usage": {
    "input_tokens": 957,
    "output_tokens": 1226,
    "tools_called": ["get_weather"],
    "seconds": 24.18
  },
  "estimated_cost_usd": 0.00071
}
```

Send the `response_id` back as `previous_response_id` on the next turn. Send an
empty string to start a new conversation.

## Design notes

**Stateless.** The server holds no conversation. A serverless function has
nowhere to keep one between requests, so continuity lives in the id the client
returns. The trade-off is measurable and worth knowing: this saves *bandwidth,
not money*. Billed input tokens still grow every turn, because the provider
retrieves the history and charges for it.

**The model has no clock.** `build_system()` runs per request and writes today's
date into the prompt, so "next month" resolves and a trip cannot be planned into
a month that has already passed.

**Mock climate data, labelled as mock.** 18 cities, 195 month entries, mostly
southern Europe. Written as 4 seasonal profiles per city and expanded to 12
months — 64 reviewable lines instead of 192 unreviewable ones. For any city not
covered, the tool returns "No climate data on file" and the prompt forbids
stating weather without a tool result.

**Bounded tools.** At most 3 rounds and 6 calls per turn. The model's arguments
are validated before they reach Python, and a tool name that is not on the
allowlist is refused.

**`previous_response_id` is validated.** The class demo backend returns a bare
`502` for any malformed id. This one checks the format and returns `422` with a
readable message.

## Run locally

```bash
pip install -r requirements.txt
set OPENAI_API_KEY=sk-...
uvicorn api.index:app --reload
```

Open http://127.0.0.1:8000

## Tests

```bash
python test_api.py           # 7 checks, no API calls, free
python test_api.py --live    # adds 4 checks that make real calls
```

## Deploy

Vercel, from this repo. Set `OPENAI_API_KEY` as an environment variable in the
project settings before the first deploy — without it `/health` still works and
`/chat` returns `503`.

## Known limits

- Climate data is invented for the exercise. It is not a climate dataset.
- The advisor does not know prices, opening hours or availability, and is
  instructed to say so rather than guess.
- A `response_id` is an unauthenticated bearer token: anyone holding one can
  continue that conversation. Acceptable for a student project, not for
  production.
