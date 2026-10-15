"""FastAPI backend for the AI Travel Advisor.

Endpoints
    GET  /              a small chat page, so a human can use it
    POST /chat          the API Luis asked for
    GET  /destinations  which cities have climate data
    GET  /health        liveness, model and coverage

Stateless by design. The server keeps no conversation: the client sends back the
response_id it received, exactly as the class demo backend does.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Vercel imports this file directly rather than as part of a package, so the
# relative form fails there and the absolute form fails locally. Try both.
try:
    from .advisor import MODEL, COVERED_CITIES, answer
except ImportError:                                  # pragma: no cover
    from advisor import MODEL, COVERED_CITIES, answer

app = FastAPI(title="AI Travel Advisor", version="1.0.0",
              description="Plans a trip through conversation. Mini-project, Ironhack AI FT Sept 26.")


class StripVercelPrefix:
    """Deliver /api/index/health to FastAPI as /health.

    vercel.json rewrites every request to /api/index, and the runtime passes that
    rewritten path through to the ASGI app. Without this, every route 404s in
    production while working perfectly on localhost.
    """

    PREFIXES = ("/api/index", "/api")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            path = scope.get("path", "")
            for prefix in self.PREFIXES:
                if path == prefix or path.startswith(prefix + "/"):
                    scope = {**scope, "path": path[len(prefix):] or "/"}
                    break
        await self.app(scope, receive, send)


app.add_middleware(StripVercelPrefix)


class ChatRequest(BaseModel):
    message: str = Field(..., description="What the traveller said")
    previous_response_id: str | None = Field(
        default="", description="The response_id from the previous turn, or empty to start fresh")


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    tools_called: list[str]
    seconds: float


class ChatResponse(BaseModel):
    answer: str
    response_id: str
    usage: Usage
    estimated_cost_usd: float


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return answer(request.message, request.previous_response_id or None)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception:
        # Never leak upstream error bodies: they can contain request details.
        raise HTTPException(status_code=502, detail="The AI provider could not process the request.")


@app.get("/destinations")
def destinations():
    return {"count": len(COVERED_CITIES), "cities": COVERED_CITIES,
            "note": "Mock climate data. Other cities work, but the advisor will not discuss "
                    "their weather."}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "cities_with_climate": len(COVERED_CITIES)}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Travel Advisor</title>
<style>
  :root { --bg:#faf9f7; --card:#fff; --ink:#1a1a1a; --muted:#6b6b6b;
          --line:#e4e2dd; --accent:#1d7a54; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#17181a; --card:#202124; --ink:#ececec; --muted:#9a9a9a;
            --line:#32343a; --accent:#58b894; }
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 system-ui, sans-serif }
  .wrap { max-width:760px; margin:0 auto; padding:24px 16px 120px }
  h1 { font-size:21px; font-weight:600; margin:0 0 4px }
  .sub { color:var(--muted); font-size:13.5px; margin-bottom:20px }
  .msg { background:var(--card); border:1px solid var(--line); border-radius:12px;
         padding:12px 15px; margin-bottom:10px; white-space:pre-wrap; overflow-wrap:anywhere }
  .me { background:transparent; border-style:dashed }
  .who { font-size:11.5px; color:var(--muted); text-transform:uppercase;
         letter-spacing:.05em; margin-bottom:5px }
  .meta { font-size:12px; color:var(--muted); margin:-4px 0 12px 2px }
  .bar { position:fixed; left:0; right:0; bottom:0; background:var(--bg);
         border-top:1px solid var(--line); padding:12px 16px }
  .bar div { max-width:760px; margin:0 auto; display:flex; gap:8px }
  input { flex:1; padding:11px 13px; border:1px solid var(--line); border-radius:9px;
          background:var(--card); color:var(--ink); font-size:15px }
  button { padding:11px 20px; border:0; border-radius:9px; background:var(--accent);
           color:#fff; font-size:15px; cursor:pointer }
  button:disabled { opacity:.5; cursor:default }
  table { border-collapse:collapse; margin:8px 0 }
  td, th { border:1px solid var(--line); padding:5px 9px; font-size:14px }
  code { background:var(--bg); padding:1px 5px; border-radius:4px; font-size:13.5px }
</style></head><body>
<div class="wrap">
  <h1>AI Travel Advisor</h1>
  <div class="sub">Tell it where you want to go. It asks what it needs, then writes the plan.
    Climate data is mock, and covers southern Europe.</div>
  <div id="log"></div>
</div>
<div class="bar"><div>
  <input id="box" placeholder="I want 4 days in Seville in August..." autofocus>
  <button id="go">Send</button>
</div></div>
<script>
let prev = "";
const log = document.getElementById("log");
const box = document.getElementById("box");
const go  = document.getElementById("go");

function md(t) {
  return t
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/^### (.*)$/gm,"<b>$1</b>")
    .replace(/^## (.*)$/gm,"<b>$1</b>")
    .replace(/^# (.*)$/gm,"<b style='font-size:18px'>$1</b>")
    .replace(/\\*\\*(.+?)\\*\\*/g,"<b>$1</b>");
}
function add(who, text, cls) {
  const d = document.createElement("div");
  d.className = "msg " + (cls || "");
  d.innerHTML = "<div class='who'>" + who + "</div>" + md(text);
  log.appendChild(d);
  window.scrollTo(0, document.body.scrollHeight);
  return d;
}
async function send() {
  const text = box.value.trim();
  if (!text) return;
  box.value = ""; go.disabled = true;
  add("you", text, "me");
  const pending = add("advisor", "thinking...");

  // A full itinerary takes 20-40 seconds. Without a visible counter the page
  // looks frozen and people reload, which starts the whole thing again.
  const t0 = Date.now();
  const tick = setInterval(() => {
    const s = Math.round((Date.now() - t0) / 1000);
    pending.innerHTML = "<div class='who'>advisor</div>thinking... " + s + "s"
      + (s > 12 ? "  <span style='color:var(--muted)'>(writing an itinerary takes a while)</span>" : "");
  }, 1000);

  try {
    const r = await fetch("/chat", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ message: text, previous_response_id: prev })
    });
    clearInterval(tick);
    const data = await r.json();
    if (!r.ok) { pending.innerHTML = "<div class='who'>error " + r.status + "</div>" + md(data.detail || "failed"); }
    else {
      prev = data.response_id;
      pending.innerHTML = "<div class='who'>advisor</div>" + md(data.answer);
      const u = data.usage;
      const m = document.createElement("div");
      m.className = "meta";
      m.textContent = u.input_tokens + " in + " + u.output_tokens + " out tokens - $"
        + data.estimated_cost_usd.toFixed(5) + " - " + u.seconds + "s"
        + (u.tools_called.length ? " - called " + u.tools_called.join(", ") : "");
      log.appendChild(m);
    }
  } catch (e) {
    clearInterval(tick);
    const s = Math.round((Date.now() - t0) / 1000);
    pending.innerHTML = "<div class='who'>error</div>No reply after " + s
      + "s. The server may have timed out. Try a shorter request.";
  }
  go.disabled = false; box.focus();
  window.scrollTo(0, document.body.scrollHeight);
}
go.onclick = send;
box.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
