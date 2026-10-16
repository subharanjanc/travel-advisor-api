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
from openai import APITimeoutError
from pydantic import BaseModel, Field

# Vercel imports this file directly rather than as part of a package, so the
# relative form fails there and the absolute form fails locally. Try both.
try:
    from .advisor import (MODEL, IMAGE_MODEL, SEASONS, WELCOME, COVERED_CITIES,
                          answer, generate_image)
except ImportError:                                  # pragma: no cover
    from advisor import (MODEL, IMAGE_MODEL, SEASONS, WELCOME, COVERED_CITIES,
                         answer, generate_image)

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


class ImageRequest(BaseModel):
    city: str = Field(..., description="City to picture, e.g. Malaga")
    season: str = Field(..., description="One of: " + ", ".join(SEASONS))


class ChatResponse(BaseModel):
    answer: str
    response_id: str
    image_request: ImageRequest | None = Field(
        default=None,
        description="When set, POST this to /image to get the picture the advisor promised.")
    usage: Usage
    estimated_cost_usd: float


class ImageResponse(BaseModel):
    city: str
    season: str
    image_base64: str
    seconds: float
    note: str


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


@app.post("/image", response_model=ImageResponse)
def image(request: ImageRequest):
    """Generate one picture. Separate from /chat because it takes about 57 seconds.

    Doing this inside a chat turn would exceed the platform's 60s function limit.
    """
    try:
        return generate_image(request.city, request.season)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except APITimeoutError:
        # Distinguished from the generic case on purpose: a timeout here was
        # once reported as "the provider could not be reached", which sent the
        # debugging in exactly the wrong direction.
        raise HTTPException(status_code=504,
                            detail="The image took too long to generate. Please try again.")
    except Exception as error:
        raise HTTPException(status_code=502,
                            detail=f"Image generation failed ({type(error).__name__}).")


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
  /* One committed green scheme, darkest to brightest:
     deep bottle -> bottle -> forest -> emerald -> turquoise -> fluorescent.
     The fluorescent is deliberately rationed: it marks Send and the live
     counters only. Spread over a whole page it stops reading as emphasis
     and starts hurting to look at. */
  :root {
    --deep:#04170F;        /* deep bottle green, the page */
    --bottle:#0A2E22;      /* bottle green, cards */
    --bottle-hi:#0E3A2B;   /* bottle green, raised */
    --forest:#16553E;      /* forest, hairlines */
    --emerald:#0FB882;     /* emerald, primary accent */
    --turq:#2EE6C5;        /* turquoise, secondary accent */
    --turq-soft:#7FE7DC;   /* pale turquoise */
    --fluoro:#5CFF4D;      /* fluorescent green, rationed */
    --mint:#E6FFF4;        /* mint white, body text */

    --bg:var(--deep); --card:var(--bottle); --ink:var(--mint);
    --muted:#74C8AE; --line:var(--forest); --accent:var(--emerald);
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.6 system-ui, sans-serif }
  .wrap { max-width:760px; margin:0 auto; padding:24px 16px 120px }
  h1 { font-size:21px; font-weight:600; margin:0 0 4px; color:var(--mint) }
  h1 span { color:var(--fluoro) }
  .sub { color:var(--muted); font-size:13.5px; margin-bottom:20px }
  .swatches { display:flex; gap:5px; margin:0 0 18px }
  .swatches i { width:34px; height:5px; border-radius:3px; display:block }
  .msg { background:var(--card); border:1px solid var(--line); border-radius:12px;
         padding:12px 15px; margin-bottom:10px; white-space:pre-wrap;
         overflow-wrap:anywhere }
  .msg b { color:var(--turq) }
  .me { background:transparent; border:1px dashed var(--emerald) }
  .me .who { color:var(--turq-soft) }
  .who { font-size:11.5px; color:var(--muted); text-transform:uppercase;
         letter-spacing:.06em; margin-bottom:5px }
  .meta { font-size:12px; color:var(--muted); margin:-4px 0 12px 2px;
          border-left:2px solid var(--forest); padding-left:8px }
  .bar { position:fixed; left:0; right:0; bottom:0; background:var(--deep);
         border-top:1px solid var(--forest); padding:12px 16px }
  .bar div { max-width:760px; margin:0 auto; display:flex; gap:8px }
  input { flex:1; padding:11px 13px; border:1px solid var(--forest); border-radius:9px;
          background:var(--bottle-hi); color:var(--ink); font-size:15px }
  input::placeholder { color:#4E9C83 }
  input:focus { outline:0; border-color:var(--turq);
                box-shadow:0 0 0 3px rgba(46,230,197,.18) }
  button { padding:11px 22px; border:0; border-radius:9px; background:var(--fluoro);
           color:var(--deep); font-size:15px; font-weight:600; cursor:pointer }
  button:hover { background:var(--turq) }
  button:disabled { background:var(--forest); color:var(--muted); cursor:default }
  table { border-collapse:collapse; margin:8px 0 }
  td, th { border:1px solid var(--forest); padding:5px 9px; font-size:14px }
  th { background:var(--bottle-hi); color:var(--turq) }
  code { background:var(--bottle-hi); color:var(--turq); padding:1px 5px;
         border-radius:4px; font-size:13.5px }
  img { border:1px solid var(--forest) }
</style></head><body>
<div class="wrap">
  <h1>AI Travel Advisor<span>.</span></h1>
  <div class="sub">Tell it where you want to go. It asks what it needs, then writes the plan
    and draws you the place.</div>
  <div class="swatches" aria-hidden="true">
    <i style="background:#0A2E22"></i><i style="background:#16553E"></i>
    <i style="background:#0FB882"></i><i style="background:#2EE6C5"></i>
    <i style="background:#7FE7DC"></i><i style="background:#5CFF4D"></i>
  </div>
  <div id="log"><div class="msg"><div class="who">advisor</div><!--WELCOME--></div></div>
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
// The picture is a second request on purpose: generating it takes about 57
// seconds, and the chat turn plus the image would exceed the function limit.
async function fetchImage(req) {
  const slot = add("picture", "drawing " + req.city + " in " + req.season + "... 0s");
  const t0 = Date.now();
  const tick = setInterval(() => {
    const s = Math.round((Date.now() - t0) / 1000);
    slot.innerHTML = "<div class='who'>picture</div>drawing " + req.city + " in "
      + req.season + "... " + s + "s <span style='color:var(--muted)'>(about a minute)</span>";
  }, 1000);
  try {
    const r = await fetch("/image", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(req)
    });
    clearInterval(tick);
    const d = await r.json();
    if (!r.ok) { slot.innerHTML = "<div class='who'>picture failed</div>" + (d.detail || r.status); return; }
    slot.innerHTML = "<div class='who'>" + d.city + ", " + d.season + "</div>"
      + "<img src='data:image/png;base64," + d.image_base64 + "' "
      + "style='width:100%;max-width:420px;border-radius:9px;display:block'>"
      + "<div style='font-size:12px;color:var(--muted);margin-top:6px'>" + d.note
      + " Generated in " + d.seconds + "s.</div>";
  } catch (e) {
    clearInterval(tick);
    slot.innerHTML = "<div class='who'>picture failed</div>The image request timed out.";
  }
  window.scrollTo(0, document.body.scrollHeight);
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
      if (data.image_request) fetchImage(data.image_request);
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
    """The welcome is rendered statically from the same constant the prompt quotes.

    Two reasons it is not a model call: it costs nothing and appears instantly,
    and the prompt tells the advisor the traveller has already read it, so nobody
    gets greeted twice.
    """
    safe = WELCOME.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br>")
    return PAGE.replace("<!--WELCOME-->", safe)
