"""Local checks before deploying. Run: python test_api.py

The first four tests cost nothing. The last one makes two real API calls.
"""

import sys
from pathlib import Path

# This machine's Python is the embeddable build, whose python314._pth stops the
# script's own folder from being added to sys.path. Put it back.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient            # noqa: E402

from api.index import app                            # noqa: E402

client = TestClient(app)
passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print("  PASS  %s" % name)
    else:
        failed += 1
        print("  FAIL  %s %s" % (name, detail))


print("\nFree checks")
r = client.get("/health")
check("health is 200", r.status_code == 200, r.text[:120])
check("health reports cities", r.json().get("cities_with_climate", 0) >= 18, r.text[:120])

r = client.get("/destinations")
check("destinations lists seville", "seville" in r.json()["cities"])

r = client.get("/")
check("home page renders", r.status_code == 200 and "<title>" in r.text)

r = client.post("/chat", json={"message": "", "previous_response_id": ""})
check("empty message is 422", r.status_code == 422, r.text[:120])

r = client.post("/chat", json={"message": "hi", "previous_response_id": "not-an-id"})
check("bad id is 422, not 502", r.status_code == 422, r.text[:160])

r = client.post("/chat", json={"message": "x" * 3000, "previous_response_id": ""})
check("over-long message is 422", r.status_code == 422, r.text[:120])

if "--live" not in sys.argv:
    print("\n%d passed, %d failed  (run with --live to also test real calls)" % (passed, failed))
    sys.exit(1 if failed else 0)

print("\nLive checks (these cost money)")
r = client.post("/chat", json={"message": "I want 4 days in Seville in August. I like food.",
                               "previous_response_id": ""})
check("first turn is 200", r.status_code == 200, r.text[:200])
first = r.json()
check("returns a response_id", first.get("response_id", "").startswith("resp_"))
check("reports usage", first["usage"]["input_tokens"] > 0)
print("       answer:", first["answer"][:150].replace("\n", " "))
print("       usage :", first["usage"], "cost $%.5f" % first["estimated_cost_usd"])

r = client.post("/chat", json={"message": "What city did I say?",
                               "previous_response_id": first["response_id"]})
second = r.json()
check("memory works across turns", "eville" in second.get("answer", ""),
      second.get("answer", "")[:120])
print("       answer:", second["answer"][:150].replace("\n", " "))

print("\n%d passed, %d failed" % (passed, failed))
sys.exit(1 if failed else 0)
