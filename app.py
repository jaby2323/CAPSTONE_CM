"""
Web front-end for the Trending Topic & Viral Signal Forecaster.

A single-file FastAPI app that:
  * serves a React UI (React is vendored locally in ./static -- no CDN, no Node
    build step, no in-browser Babel; works fully offline)
  * exposes /process, which runs the SAME pipeline the notebook runs
    (the ForecastController) and streams live per-agent progress to the
    browser via Server-Sent Events (SSE).

Run it with:
    python app.py
A browser opens automatically at http://127.0.0.1:8000 -- click "Process".
"""

import os
import json
import queue
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, Response

from vector_store import VectorStore
from agents import ForecastController

app = FastAPI()

_STATIC = os.path.join(os.path.dirname(__file__), "static")


def _read_static(name):
    with open(os.path.join(_STATIC, name), "r", encoding="utf-8") as f:
        return f.read()


# Load the vendored React bundles once at startup.
REACT_JS = _read_static("react.production.min.js")
REACTDOM_JS = _read_static("react-dom.production.min.js")


@app.get("/static/react.js")
def _serve_react():
    return Response(REACT_JS, media_type="application/javascript")


@app.get("/static/react-dom.js")
def _serve_reactdom():
    return Response(REACTDOM_JS, media_type="application/javascript")


# --- Streaming endpoint: runs the pipeline, emits live agent progress --------
@app.get("/process")
def process():
    def event_stream():
        q = queue.Queue()

        # Each agent reports running/done here; pushed straight to the browser.
        def on_event(agent, state):
            q.put({"type": "status", "agent": agent, "state": state})

        def worker():
            try:
                store = VectorStore()          # long-term memory (persists)
                store.seed_demo_data()         # seed history on first run
                controller = ForecastController(store=store)
                result = controller.run_cycle(verbose=False, on_event=on_event)
                q.put({"type": "result", "data": result})
            except Exception as e:             # noqa: BLE001
                q.put({"type": "error", "message": str(e)})
            finally:
                q.put({"type": "done"})

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = q.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["type"] == "done":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- The React single-page UI ------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Viral Signal Forecaster</title>
<link rel="icon" href="data:,"/>
<script src="/static/react.js"></script>
<script src="/static/react-dom.js"></script>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         background: #0f1117; color: #e6e8ee; }
  header { display: flex; align-items: center; justify-content: space-between;
           padding: 16px 24px; border-bottom: 1px solid #232735; background: #151823; }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  .btn { background: #4f7cff; color: #fff; border: none; padding: 10px 22px;
         font-size: 15px; border-radius: 8px; cursor: pointer; font-weight: 600; }
  .btn:disabled { background: #3a3f52; cursor: not-allowed; }
  .layout { display: flex; min-height: calc(100vh - 65px); }
  .sidebar { width: 260px; padding: 20px; border-right: 1px solid #232735;
             background: #12151f; }
  .sidebar h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
                color: #8b90a3; margin: 0 0 14px; }
  .agent { display: flex; align-items: center; gap: 10px; padding: 12px 14px;
           border-radius: 8px; margin-bottom: 8px; background: #1a1e2b; }
  .agent .name { font-size: 14px; font-weight: 500; }
  .agent .sub { font-size: 11px; color: #8b90a3; }
  .badge { width: 22px; height: 22px; border-radius: 50%; display: flex;
           align-items: center; justify-content: center; font-size: 13px; flex: none; }
  .idle { background: #2a2f3f; color: #6b7080; }
  .running { background: #c9a227; color: #0f1117; animation: pulse 1s infinite; }
  .done { background: #2faa5e; color: #fff; }
  .error { background: #d8504d; color: #fff; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .4; } }
  .main { flex: 1; padding: 24px 32px; }
  .empty { color: #6b7080; margin-top: 40px; text-align: center; }
  .card { background: #161a26; border: 1px solid #232735; border-radius: 12px;
          padding: 18px 20px; margin-bottom: 16px; }
  .card .top { display: flex; align-items: center; justify-content: space-between; }
  .card .topic { font-size: 16px; font-weight: 600; }
  .conf { font-size: 13px; color: #8b90a3; margin-top: 2px; }
  .status { font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 20px; }
  .RELEASED { background: #15402a; color: #4fdc8a; }
  .HELD_FOR_REVIEW { background: #4a3410; color: #f2c14e; }
  .reasons { font-size: 12px; color: #f2c14e; margin-top: 8px; }
  .brief { margin-top: 12px; font-size: 14px; line-height: 1.55; color: #cfd3df;
           white-space: pre-wrap; }
  .bar { height: 6px; background: #2a2f3f; border-radius: 4px; margin-top: 10px;
         overflow: hidden; }
  .bar > div { height: 100%; background: linear-gradient(90deg,#4f7cff,#4fdc8a); }
</style>
</head>
<body>
<div id="root"></div>
<script>
// Plain React (no JSX / no Babel). h() is the React.createElement shorthand.
var useState = React.useState;
var h = React.createElement;

var AGENTS = [
  { key: "collector", name: "Collector Agent", sub: "fetch live data" },
  { key: "retrieval", name: "Retrieval Agent", sub: "semantic memory (RAG)" },
  { key: "critic",    name: "Critic Agent",    sub: "Tree-of-Thought scoring" },
  { key: "forecast",  name: "Forecast Agent",  sub: "briefing + write-back" }
];

function symbol(state) {
  if (state === "done") return "✓";    // check mark
  if (state === "running") return "⋯";  // ...
  if (state === "error") return "✕";    // x
  return "";
}

// Minimal markdown: **bold** -> <strong>.
function renderBrief(text) {
  var parts = (text || "").split(/(\*\*[^*]+\*\*)/g);
  return parts.map(function (p, i) {
    if (p.indexOf("**") === 0 && p.lastIndexOf("**") === p.length - 2) {
      return h("strong", { key: i }, p.slice(2, -2));
    }
    return h("span", { key: i }, p);
  });
}

function App() {
  var s = useState({});      var statuses = s[0], setStatuses = s[1];
  var r = useState(null);    var results = r[0],  setResults = r[1];
  var n = useState(false);   var running = n[0],  setRunning = n[1];
  var e = useState(null);    var error = e[0],    setError = e[1];

  function process() {
    setRunning(true); setResults(null); setError(null);
    var reset = {}; AGENTS.forEach(function (a) { reset[a.key] = "idle"; });
    setStatuses(reset);

    var es = new EventSource("/process");
    es.onmessage = function (ev) {
      var msg = JSON.parse(ev.data);
      if (msg.type === "status") {
        setStatuses(function (prev) {
          var copy = Object.assign({}, prev); copy[msg.agent] = msg.state; return copy;
        });
      } else if (msg.type === "result") {
        setResults(msg.data.ok ? msg.data.results : []);
        if (!msg.data.ok) setError("Pipeline could not produce results.");
      } else if (msg.type === "error") {
        setError(msg.message);
      } else if (msg.type === "done") {
        es.close(); setRunning(false);
      }
    };
    es.onerror = function () { es.close(); setRunning(false);
                              setError("Connection lost."); };
  }

  var sidebar = h("aside", { className: "sidebar" },
    h("h2", null, "Agents"),
    AGENTS.map(function (a) {
      var st = statuses[a.key] || "idle";
      return h("div", { className: "agent", key: a.key },
        h("div", { className: "badge " + st }, symbol(st)),
        h("div", null,
          h("div", { className: "name" }, a.name),
          h("div", { className: "sub" }, a.sub)
        )
      );
    })
  );

  var cards;
  if (error) {
    cards = h("div", { className: "empty" }, "⚠️ " + error);
  } else if (!results) {
    cards = h("div", { className: "empty" }, "Click ", h("b", null, "Process"),
             " to run a forecast cycle.");
  } else if (results.length === 0) {
    cards = h("div", { className: "empty" }, "No forecasts produced this cycle.");
  } else {
    cards = results.map(function (rc, i) {
      return h("div", { className: "card", key: i },
        h("div", { className: "top" },
          h("div", null,
            h("div", { className: "topic" }, rc.topic),
            h("div", { className: "conf" },
              "Confidence: " + (rc.confidence * 100).toFixed(0) + "%")
          ),
          h("span", { className: "status " + rc.status },
            rc.status.replace(/_/g, " "))
        ),
        h("div", { className: "bar" },
          h("div", { style: { width: (rc.confidence * 100) + "%" } })),
        (rc.review && rc.review.requires_review)
          ? h("div", { className: "reasons" },
              "⚑ Review: " + rc.review.reasons.join(", "))
          : null,
        h("div", { className: "brief" }, renderBrief(rc.briefing))
      );
    });
  }

  return h("div", null,
    h("header", null,
      h("h1", null, "📈 Trending Topic & Viral Signal Forecaster"),
      h("button", { className: "btn", onClick: process, disabled: running },
        running ? "Processing…" : "Process")
    ),
    h("div", { className: "layout" }, sidebar, h("main", { className: "main" }, cards))
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(h(App));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import webbrowser

    URL = "http://127.0.0.1:8000"
    print("\n" + "=" * 50)
    print("  Viral Signal Forecaster is starting...")
    print(f"  Open this in your browser:  {URL}")
    print("  (Press CTRL+C here to stop the server.)")
    print("=" * 50 + "\n")

    threading.Timer(1.5, lambda: webbrowser.open(URL)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
