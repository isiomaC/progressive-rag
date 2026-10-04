#!/usr/bin/env python
"""Builds site/index.html from experiments/results.json.

The output is a single self-contained HTML file (no CDN, no external JS):
charts are hand-rolled SVG rendered from the embedded JSON, so the page
works offline and always matches the committed results.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragprog.config import from_env  # noqa: E402

PHASE_ORDER = ["1-naive", "2-chunking", "3-retrieval", "6-advanced", "6-graph", "6-agentic", "6-selfrag"]
PHASE_LABEL = {
    "1-naive": "1 · Naive",
    "2-chunking": "2 · Chunking",
    "3-retrieval": "3 · Advanced retrieval",
    "6-advanced": "6.1 · Advanced combo",
    "6-graph": "6.2 · GraphRAG",
    "6-agentic": "6.3 · Agentic",
    "6-selfrag": "6.4 · Self-RAG",
}
PHASE_COLOR = {
    "1-naive": "#8b949e", "2-chunking": "#d29922", "3-retrieval": "#58a6ff",
    "6-advanced": "#a371f7", "6-graph": "#3fb950", "6-agentic": "#f778ba",
    "6-selfrag": "#ff7b72",
}

CSS = """
:root {
  --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
  --muted:#8b949e; --accent:#58a6ff;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font:15px/1.55 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif; }
header { padding:56px 32px 24px; max-width:1180px; margin:0 auto; }
h1 { font-size:34px; margin:0 0 8px; letter-spacing:-0.5px; }
h1 .grad { background:linear-gradient(90deg,#58a6ff,#a371f7,#f778ba);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
h2 { font-size:22px; margin:48px 0 14px; letter-spacing:-0.3px; }
h3 { margin:0 0 10px; color:var(--muted); font-weight:600;
  text-transform:uppercase; letter-spacing:0.8px; font-size:12px; }
.sub { color:var(--muted); font-size:16px; max-width:760px; }
main { max-width:1180px; margin:0 auto; padding:0 32px 80px; }
.badges { display:flex; gap:8px; flex-wrap:wrap; margin-top:16px; }
.badge { border:1px solid var(--border); border-radius:999px; padding:4px 12px;
  font-size:12.5px; color:var(--muted); background:var(--panel); }
.badge b { color:var(--text); font-weight:600; }
.panel { background:var(--panel); border:1px solid var(--border);
  border-radius:12px; padding:20px 22px; margin:14px 0; }
.grid { display:grid; gap:14px; }
.grid.cards { grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }
.grid.split { grid-template-columns:1.4fr 1fr; }
@media (max-width:900px){ .grid.split { grid-template-columns:1fr; } }
.card { background:var(--panel); border:1px solid var(--border);
  border-radius:12px; padding:18px 20px; }
.card .tag { font-size:11px; text-transform:uppercase; letter-spacing:1px;
  color:var(--muted); font-weight:600; }
.card h4 { margin:6px 0 8px; font-size:17px; }
.card p { color:var(--muted); font-size:13.5px; margin:8px 0 0; }
.card .big { font-size:26px; font-weight:700; margin-top:10px;
  font-variant-numeric:tabular-nums; }
.card .delta { font-size:13px; font-weight:600; }
.up { color:#3fb950; } .down { color:#ff7b72; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { color:var(--muted); text-align:right; font-weight:600; padding:7px 9px;
  border-bottom:1px solid var(--border); white-space:nowrap; font-size:11.5px;
  text-transform:uppercase; letter-spacing:0.5px; }
th:first-child, td:first-child { text-align:left; }
td { padding:7px 9px; border-bottom:1px solid #21262d;
  text-align:right; font-variant-numeric:tabular-nums; }
tr:hover td { background:#1c2128; }
td.vname { white-space:nowrap; }
td.win { color:#3fb950; font-weight:700; }
.phase-pill { display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:8px; }
.stepper { display:flex; gap:6px; flex-wrap:wrap; }
.step { border:1px solid var(--border); border-radius:10px; padding:10px 14px;
  flex:1; min-width:130px; background:var(--panel); }
.step .t { font-size:12px; color:var(--muted); }
.step .n { font-size:15px; font-weight:700; margin-top:4px;
  font-variant-numeric:tabular-nums; }
svg { width:100%; height:auto; display:block; }
svg text { font-family:inherit; fill:var(--muted); font-size:11px; }
.chart-title { font-size:13px; color:var(--text); font-weight:600; margin-bottom:6px; }
.legend { display:flex; gap:16px; flex-wrap:wrap; margin-top:10px; font-size:12px;
  color:var(--muted); }
.legend i { display:inline-block; width:10px; height:10px; border-radius:3px;
  margin-right:5px; }
.heatmap { overflow-x:auto; }
.heatmap svg { min-width:640px; }
.note { font-size:12.5px; color:var(--muted); margin-top:10px; }
code { background:#21262d; border-radius:5px; padding:1px 6px;
  font-size:12.5px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
footer { color:var(--muted); font-size:12.5px; padding:30px 32px 60px;
  max-width:1180px; margin:0 auto; }
a { color:var(--accent); text-decoration:none; }
"""


def build_report(cfg) -> None:
    results = json.loads(cfg.results_file.read_text())
    data_json = json.dumps(results).replace("</", "<\\/")
    # model comparison: any results-*.json siblings (e.g. offline/deepseek/openai)
    compares = []
    for p in sorted(cfg.results_file.parent.glob("results-*.json")):
        try:
            c = json.loads(p.read_text())
        except Exception:
            continue
        compares.append({"file": p.name, "meta": c["meta"], "variants": c["variants"]})
    compare_json = json.dumps(compares).replace("</", "<\\/")
    phase_json = json.dumps(PHASE_ORDER)
    label_json = json.dumps(PHASE_LABEL)
    color_json = json.dumps(PHASE_COLOR)
    html = _page(css=CSS, data=data_json, compare=compare_json, phase_order=phase_json,
                 phase_labels=label_json, phase_colors=color_json)
    cfg.site_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.site_dir / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html) // 1024} KB)")


def _page(css: str, data: str, compare: str, phase_order: str, phase_labels: str, phase_colors: str) -> str:
    # NOTE: this is an f-string; JS braces are escaped as {{ / }} and the only
    # Python fields are the {css}/{data}/{phase_*} substitutions above.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Progressive RAG — Experiments &amp; Learning Report</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>Progressive <span class="grad">RAG</span> Documentation Assistant</h1>
  <p class="sub">A learning-first project: start from the simplest possible RAG and upgrade it
  step by step — Naive → Chunking → Advanced Retrieval → Graph → Agentic → Corrective/Self-RAG —
  measuring every change against a fixed golden set.</p>
  <div class="badges" id="badges"></div>
</header>
<main>
  <h2>The journey at a glance</h2>
  <div class="stepper" id="stepper"></div>

  <h2>Model comparison <span style="color:var(--muted);font-size:13px;">— same harness, different generators</span></h2>
  <div class="panel" id="model-compare"></div>

  <h2>Learning highlights</h2>
  <div class="grid cards" id="highlights"></div>

  <h2>Chunking experiment <span style="color:var(--muted);font-size:13px;">— Phase 2: the highest-leverage knob</span></h2>
  <div class="grid split">
    <div class="panel"><div class="chart-title">Answer correctness, chunking strategies (same vector retriever)</div><div id="chart-chunk"></div></div>
    <div class="panel"><div class="chart-title">Why chunking wins</div><p id="chunk-why" class="note"></p></div>
  </div>

  <h2>Advanced retrieval <span style="color:var(--muted);font-size:13px;">— Phase 3: what each upgrade buys</span></h2>
  <div class="panel"><div class="chart-title">Document recall@5 and answer correctness per retriever</div><div id="chart-retr"></div>
  <div class="legend" id="legend-retr"></div></div>

  <h2>Latency &amp; cost <span style="color:var(--muted);font-size:13px;">— Phase 5: what each technique actually costs</span></h2>
  <div class="grid split">
    <div class="panel"><div class="chart-title">Median latency per variant: retrieval + generation (ms)</div><div id="chart-lat"></div></div>
    <div class="panel"><h3>Per-component timing</h3><div id="comp-lat"></div></div>
  </div>
  <div class="panel" id="cost-panel"></div>

  <h2>Multi-hop questions <span style="color:var(--muted);font-size:13px;">— where single-shot retrieval breaks</span></h2>
  <div class="grid split">
    <div class="panel"><div class="chart-title">Multi-hop complete rate — fraction of multi-hop questions where ALL required docs made the top-5</div><div id="chart-mh"></div></div>
    <div class="panel"><div class="chart-title">Single-hop vs multi-hop, baseline</div><div id="chart-mhgap"></div></div>
  </div>

  <h2>GraphRAG &amp; Agentic patterns <span style="color:var(--muted);font-size:13px;">— Phase 6: when do they actually help?</span></h2>
  <div class="grid cards" id="phase6cards"></div>

  <h2>Corrective / Self-RAG <span style="color:var(--muted);font-size:13px;">— Phase 6.4: honesty as a feature</span></h2>
  <div class="panel" id="selfrag-panel"></div>

  <h2>Practical challenges lab <span style="color:var(--muted);font-size:13px;">— Phase 5</span></h2>
  <div class="grid cards" id="challenge-cards"></div>

  <h2>Knowledge-base updates</h2>
  <div class="panel" id="kb-panel"></div>

  <h2>Full comparison table</h2>
  <div class="panel" style="overflow-x:auto"><table id="full-table"></table></div>

  <h2>Question-level heatmap <span style="color:var(--muted);font-size:13px;">— answer correctness, rows = questions, columns = variants</span></h2>
  <div class="panel heatmap" id="heatmap"></div>

  <h2>Methodology</h2>
  <div class="panel" id="method"></div>
</main>
<footer id="footer"></footer>
<script>
const R = {data};
const COMPARES = {compare};
const PHASE_ORDER = {phase_order};
const PHASE_LABELS = {phase_labels};
const PHASE_COLORS = {phase_colors};
const V = R.variants;
const byName = Object.fromEntries(V.map(v => [v.name, v]));
const phaseOf = v => PHASE_ORDER.indexOf(v.phase);
const phColor = v => PHASE_COLORS[v.phase] || "#8b949e";
const m = (v, key) => (v.metrics.overall[key] ?? 0);
const byType = (v, type, key) => {{ const t = v.metrics.by_type[type]; return t ? (t[key] ?? 0) : 0; }};
function sortedV() {{ return [...V].sort((a,b) => phaseOf(a)-phaseOf(b) || a.name.localeCompare(b.name)); }}
const pct = x => (x === null || x === undefined) ? "—" : (x * 100).toFixed(0) + "%";

// ---------- badges ----------
const meta = R.meta;
document.getElementById("badges").innerHTML = [
  `<span class="badge">corpus <b>${{meta.corpus_docs}}</b> docs · <b>${{meta.corpus_words.toLocaleString()}}</b> words</span>`,
  `<span class="badge">golden set <b>${{meta.golden_questions}}</b> questions</span>`,
  `<span class="badge">retriever k = <b>${{meta.k}}</b></span>`,
  `<span class="badge">embedder <b>${{meta.embedder}}</b> ${{meta.embedder_semantic ? "(semantic)" : "(lexical)"}}</span>`,
  `<span class="badge">LLM <b>${{meta.llm}}</b> ${{meta.llm_offline ? "(offline demo mode)" : "(live API)"}}</span>`,
  `<span class="badge">runtime <b>${{R.total_runtime_s}}s</b></span>`,
  `<span class="badge">generated <b>${{meta.generated_at}}</b></span>`,
].join("");

// ---------- stepper ----------
const best = new Map();
for (const v of V) {{
  const cur = best.get(v.phase);
  if (!cur || m(v, "answer_correctness") > m(cur, "answer_correctness")) best.set(v.phase, v);
}}
document.getElementById("stepper").innerHTML = PHASE_ORDER.map(p => {{
  const v = best.get(p);
  if (!v) return "";
  return `<div class="step">
    <div class="t">${{PHASE_LABELS[p]}}</div>
    <div class="n">${{v.name}}</div>
    <div class="t">correctness ${{(m(v,"answer_correctness")*100).toFixed(1)}}%</div>
  </div>`;
}}).join("");

// ---------- model comparison ----------
(function() {{
  const runs = [{{ label: meta.llm === "offline" ? "offline (extractive)" : meta.llm, meta, variants: V }}]
    .concat(COMPARES.map(c => ({{ label: c.meta.llm === "offline" ? "offline (extractive)" : c.meta.llm, meta: c.meta, variants: c.variants }})));
  const el = document.getElementById("model-compare");
  if (runs.length < 2) {{
    el.innerHTML = `<p class="note">Only one run is present. Save additional runs as
      <code>experiments/results-&lt;model&gt;.json</code> (e.g. <code>--out experiments/results-deepseek.json</code>)
      and rebuild to see the model comparison.</p>`;
    return;
  }}
  const keyRows = [
    ["Answer correctness", "answer_correctness"],
    ["Doc recall@5", "doc_recall"],
    ["Hit rate", "hit_rate"],
    ["MRR", "mrr"],
    ["NDCG@5", "ndcg"],
    ["Faithfulness (LLM judge)", "faithfulness_llm"],
  ];
  let rows = keyRows.map(([label, key]) => {{
    const per = runs.map(run => {{
      const vs = run.variants;
      const present = vs.filter(v => typeof (v.metrics.overall[key]) === "number");
      if (!present.length) return null;
      const sum = present.reduce((a, v) => a + v.metrics.overall[key], 0);
      return sum / present.length;
    }});
    const idxs = per.map((x, i) => x === null ? -1 : i).filter(i => i >= 0);
    const bestIdx = idxs.length ? idxs.reduce((a, b) => per[b] > per[a] ? b : a) : -1;
    return `<tr><td class="vname">${{label}}</td>${{per.map((x,i) =>
      `<td class="${{i === bestIdx ? "win" : ""}}">${{x === null ? "—" : (x*100).toFixed(1)}}</td>`).join("")}}</tr>`;
  }}).join("");
  el.innerHTML = `
    <table>
      <tr><th>metric (mean across 16 variants)</th>${{runs.map(run =>
        `<th>${{run.label}}<br><span style="font-weight:400;font-size:10px">${{run.meta.generated_at}}</span></th>`).join("")}}</tr>
      ${{rows}}
    </table>
    <p class="note">The same 51 questions, the same retrievers, only the generator differs. The extractive
    offline mode answers by quoting retrieved sentences (faithful by construction, correctness capped by
    retrieval); live LLMs paraphrase and synthesize, which is why correctness jumps. The variant-level
    comparison below shows every configuration side by side.</p>`;
  // variant-level correctness table
  const vnames = [...new Set(runs.flatMap(run => run.variants.map(v => v.name)))];
  const vrows = vnames.map(n => {{
    const per = runs.map(run => {{
      const v = run.variants.find(x => x.name === n);
      return v ? (v.metrics.overall.answer_correctness ?? 0) : null;
    }});
    const bestIdx = per.indexOf(Math.max(...per.filter(x => x !== null)));
    return `<tr><td class="vname">${{n}}</td>${{per.map((x,i) =>
      `<td class="${{i === bestIdx ? "win" : ""}}">${{x === null ? "—" : (x*100).toFixed(1)}}</td>`).join("")}}</tr>`;
  }}).join("");
  el.insertAdjacentHTML("beforeend", `
    <div style="margin-top:18px"><div class="chart-title">Answer correctness by variant, per run</div>
    <table><tr><th>variant</th>${{runs.map(run => `<th>${{run.label}}</th>`).join("")}}</tr>${{vrows}}</table></div>`);
}})();

// ---------- highlights ----------
const base = byName["baseline"];
const bestChunkRank = [...V].filter(v => v.phase === "2-chunking").sort((a,b) => m(b,"mrr")-m(a,"mrr"))[0];
const dBest = v => (m(v,"answer_correctness") - m(base,"answer_correctness")) / Math.max(0.001, m(base,"answer_correctness"));
const rerank = byName["retr-hybrid-rerank"], hyde = byName["retr-hyde"];
const graph = byName["graph"], agentic = byName["agentic"], selfrag = byName["self-rag"], adv = byName["advanced-combo"];
const bad = R.challenges.bad_retrieval;
const kb = R.challenges.kb_update;

function card(tag, title, body, color) {{
  return `<div class="card"><div class="tag" style="color:${{color||"var(--muted)"}}">${{tag}}</div><h4>${{title}}</h4>${{body}}</div>`;
}}
const hi = [];
hi.push(card("Phase 2 · biggest lever", "Chunking matters — watch the ranking, not just the answer",
  `<div class="big">+${{((m(bestChunkRank,"mrr") - m(base,"mrr"))/Math.max(0.001,m(base,"mrr"))*100).toFixed(0)}}% <span style="font-size:14px;color:var(--muted)">MRR: ${{bestChunkRank.name}} vs fixed-size baseline</span></div>
   <p>Sentence-aware recursive chunking lifts MRR ${{(m(bestChunkRank,"mrr")*100).toFixed(0)}}% vs ${{(m(base,"mrr")*100).toFixed(0)}}% with the <i>identical</i> retriever. The mechanism is in the diagnostics: fixed windows cut ~25% of corpus sentences in half, and on the questions they damage, sentence-aware chunking wins.
   ${{meta.llm_offline ? "" : " With a live LLM the answer-correctness gap shrinks — the model infers across damaged context — which is itself the lesson: chunking buys most where the generator is weakest."}}</p>`, "#d29922"));
hi.push(card("Phase 3 · best per ms", "Reranking buys the most per millisecond",
  `<div class="big">${{pct(m(rerank,"doc_recall"))}} <span style="font-size:14px;color:var(--muted)">hybrid+rerank doc recall</span></div>
   <p>Reranking lifts retrieval with no index change — costing ${{byName["retr-rerank"].latency.retrieval_ms_median.toFixed(0)}} ms/query on CPU (cross-encoder). HyDE and multi-query are gambles: HyDE scored ${{pct(m(hyde,"doc_recall"))}} doc recall vs ${{pct(m(base,"doc_recall"))}} baseline — the LLM's guess can backfire.</p>`, "#58a6ff"));
hi.push(card("Phase 5 · silent failure", "Forced bad retrieval caps answer quality",
  `<div class="big"><span class="down">${{pct(bad.correctness_bad_mean)}}</span> <span style="font-size:14px;color:var(--muted)">correctness on worst-k chunks vs</span> <span class="up">${{pct(bad.correctness_good_mean)}}</span></div>
   <p>Generation cannot recover what retrieval never found — answer quality is capped by chunk quality. Detect it first: flag queries whose best similarity score is low <i>before</i> generating.</p>`, "#8b949e"));
hi.push(card("Phase 5 · staleness", "Knowledge-base changes break answers silently",
  `<div class="big">${{pct(kb.doc_recall_affected_mean)}} <span style="font-size:14px;color:var(--muted)">recall on affected questions after deleting 07-reranking.md</span></div>
   <p>Control questions: ${{pct(kb.doc_recall_control_mean)}} recall. Adding a new doc flipped q51 from unanswerable to retrievable. The index must be rebuilt on every KB change.</p>`, "#8b949e"));
hi.push(card("Phase 6 · targeted tools", "GraphRAG & agents help exactly where expected",
  `<div class="big">${{pct(byType(agentic,"multi-hop","doc_recall"))}} <span style="font-size:14px;color:var(--muted)">agentic multi-hop doc recall vs</span> ${{pct(byType(base,"multi-hop","doc_recall"))}} <span style="font-size:14px;color:var(--muted)">baseline</span></div>
   <p>On single-hop questions the extra loops add latency and change nothing. Agents and graphs are multi-hop / relational specialists, not default upgrades.</p>`, "#f778ba"));
hi.push(card("Phase 6.4 · honesty", "Self-RAG trades silence for hallucination",
  `<div class="big">${{pct(byType(selfrag,"tricky","faithfulness"))}} <span style="font-size:14px;color:var(--muted)">faithfulness on tricky questions</span></div>
   <p>Critiquing its own context and answering "I don't know" when support is missing keeps answers grounded — the tricky subset exists precisely to test that.</p>`, "#ff7b72"));
document.getElementById("highlights").innerHTML = hi.join("");

// ---------- chart helpers ----------
function hbar(el, items, valueFn, opts = {{}}) {{
  const W = el.clientWidth || 700, H = Math.max(60, items.length * 30 + 24);
  const padL = opts.labelWidth ?? 150, padR = 64, padT = 6;
  const maxV = opts.max ?? Math.max(...items.map(valueFn), 1e-6);
  const rows = items.map((it, i) => {{
    const v = valueFn(it);
    const x = padL, y = padT + i * 30, w = (W - padL - padR) * (v / maxV), h = 22;
    const col = opts.color ? (typeof opts.color === "function" ? opts.color(it) : opts.color) : "#58a6ff";
    return `<g>
      <text x="${{padL - 8}}" y="${{y + h - 6}}" text-anchor="end">${{opts.label(it)}}</text>
      <rect x="${{x}}" y="${{y}}" width="${{Math.max(w, 1)}}" height="${{h}}" rx="4" fill="${{col}}" opacity="0.85"></rect>
      <text x="${{x + Math.max(w, 1) + 6}}" y="${{y + h - 6}}">${{opts.valueFmt ? opts.valueFmt(v) : pct(v)}}</text>
    </g>`;
  }}).join("");
  el.innerHTML = `<svg viewBox="0 0 ${{W}} ${{H}}">${{rows}}</svg>`;
}}
function grouped(el, groups, series, opts = {{}}) {{
  const W = el.clientWidth || 700;
  const H = Math.max(90, groups.length * 34 + 30);
  const padL = 150, padR = 30, padT = 8, padB = 24;
  const maxV = opts.max ?? Math.max(...groups.flatMap((g,i) => series.map(s => s.value(g,i))), 1e-6);
  const band = (W - padL - padR) / groups.length;
  const bw = Math.min(band * 0.75 / series.length, 46);
  let out = "";
  groups.forEach((g, gi) => {{
    const cx = padL + band * gi + band / 2;
    series.forEach((s, si) => {{
      const v = s.value(g, gi);
      const x = cx - (series.length * bw) / 2 + si * bw;
      const h = Math.max(1, (H - padT - padB) * (v / maxV));
      out += `<rect x="${{x}}" y="${{H - padB - h}}" width="${{bw - 2}}" height="${{h}}" rx="3" fill="${{s.color}}" opacity="0.9"><title>${{s.name}}: ${{pct(v)}}</title></rect>`;
    }});
    out += `<text x="${{cx}}" y="${{H - 8}}" text-anchor="middle">${{opts.label ? opts.label(g) : g}}</text>`;
  }});
  el.innerHTML = `<svg viewBox="0 0 ${{W}} ${{H}}">${{out}}</svg>`;
}}

// ---------- chunking chart ----------
const chunkers = ["chunk-fixed","chunk-recursive","chunk-semantic","chunk-parent-child"].map(n => byName[n]);
hbar(document.getElementById("chart-chunk"), chunkers,
  v => m(v, "answer_correctness"),
  {{ label: v => v.name, color: "#d29922" }});
document.getElementById("chunk-why").innerHTML =
  `<b>Fixed windows cut information in half; sentence-aware chunking keeps it whole.</b> The boundary-integrity
   diagnostic measures the damage directly: fixed windows split ~25% of corpus sentences, recursive ~24%,
   semantic 0%, parent-child ~5%. A sentence cut across chunks is an answer no retriever can recover — which
   is why recursive chunking ranks better (MRR +18% here) with the identical vector retriever, and why the
   questions that fixed chunking damages score worse under fixed than under semantic chunking.
   <br><br>Two honest caveats the experiments taught us: <b>(1) chunk size must scale to corpus size</b>
   (we use 60 words on an 8k-word corpus, not the 500-word production norm), and <b>(2) strong generators
   partially repair split chunks by inferring</b> — in live-LLM mode the answer-quality gaps shrink even
   though the ranking gaps persist. Reranking fixes <i>selection</i> errors; chunking fixes
   <i>information</i> errors — they compose, which is why the strongest configurations pair them.`;

// ---------- retrieval chart ----------
const retrs = ["retr-vector","retr-bm25","retr-hybrid","retr-multi-query","retr-hyde","retr-rerank","retr-hybrid-rerank"].map(n => byName[n]).filter(Boolean);
grouped(document.getElementById("chart-retr"), retrs,
  [ {{ name:"doc recall@5", color:"#58a6ff", value:v => m(v,"doc_recall") }},
    {{ name:"answer correctness", color:"#d29922", value:v => m(v,"answer_correctness") }} ],
  {{ label: v => v.name }});
document.getElementById("legend-retr").innerHTML =
  `<span><i style="background:#58a6ff"></i>document recall@5 (did we fetch the right docs?)</span>
   <span><i style="background:#d29922"></i>answer correctness (is the final answer right?)</span>`;

// ---------- latency ----------
const latV = sortedV().filter(v => !v.phase.startsWith("2-chunking") || v.name === "chunk-recursive");
hbar(document.getElementById("chart-lat"), latV,
  v => v.latency.retrieval_ms_median + v.latency.generation_ms_median,
  {{ label: v => v.name, valueFmt: v => v.toFixed(0) + " ms", color: v => phColor(v) }});
const cl = R.challenges.latency;
document.getElementById("comp-lat").innerHTML = `
  <table>
    <tr><th>component</th><th>ms</th></tr>
    <tr><td class="vname">embed query (MiniLM)</td><td>${{cl.embed_query_ms.toFixed(2)}}</td></tr>
    <tr><td class="vname">vector search</td><td>${{cl.vector_search_ms.toFixed(2)}}</td></tr>
    <tr><td class="vname">BM25 search</td><td>${{cl.bm25_search_ms.toFixed(2)}}</td></tr>
    <tr><td class="vname">hybrid retrieval</td><td>${{cl.hybrid_retrieval_ms.toFixed(2)}}</td></tr>
    <tr><td class="vname">hybrid + cross-encoder rerank</td><td>${{cl.hybrid_plus_rerank_ms.toFixed(1)}}</td></tr>
    <tr><td class="vname">generation (${{meta.llm_offline ? "offline extractive" : "live LLM"}})</td><td>${{cl.generation_ms.toFixed(2)}}</td></tr>
  </table>
  <p class="note">${{cl.note}}</p>`;
document.getElementById("cost-panel").innerHTML = `<h3>Cost accounting ${{meta.llm_offline ? "(offline demo mode — zero API spend)" : "(live LLM)"}}</h3>
<table>
<tr><th>variant</th><th>prompt tokens</th><th>completion tokens</th><th>LLM calls</th><th>est. $</th></tr>
${{sortedV().map(v => `<tr><td class="vname"><span class="phase-pill" style="background:${{phColor(v)}}"></span>${{v.name}}</td>
<td>${{v.cost.prompt_tokens}}</td><td>${{v.cost.completion_tokens}}</td><td>${{v.cost.n_calls}}</td><td>${{v.cost.usd ? "$" + v.cost.usd.toFixed(4) : "0"}}</td></tr>`).join("")}}
</table>
<p class="note">Every "smart" retrieval technique is a loan of tokens: multi-query and HyDE spend LLM calls up front,
reranking spends compute instead of tokens, agentic loops spend both — per <i>step</i>.</p>`;

// ---------- multi-hop ----------
hbar(document.getElementById("chart-mh"), sortedV(),
  v => byType(v, "multi-hop", "multi_hop_complete"),
  {{ label: v => v.name, color: v => phColor(v) }});
const shBase = byType(base, "single-hop", "doc_recall"), mhBase = byType(base, "multi-hop", "multi_hop_complete");
document.getElementById("chart-mhgap").innerHTML = `<div class="big" style="margin:6px 0">${{pct(shBase)}} → ${{pct(mhBase)}}</div>
<p class="note">The baseline's document recall collapses from single-hop to multi-hop questions, because one
top-k pass is dominated by the most salient keyword (hop one). Parent-doc, graph and agentic variants recover
part of the gap — see the chart on the left. Golden answers list <i>every</i> required document, so a multi-hop
hit counts only when all of them are in the context.</p>`;

// ---------- phase 6 cards ----------
function deltaPct(v) {{ return (byType(v, "multi-hop", "multi_hop_complete") - mhBase) >= 0 ? "+" + ((byType(v, "multi-hop", "multi_hop_complete") - mhBase)*100).toFixed(0) + " pts" : ((byType(v, "multi-hop", "multi_hop_complete") - mhBase)*100).toFixed(0) + " pts"; }}
document.getElementById("phase6cards").innerHTML = [
  card("6.1 · Advanced combo", "Composition beats any single trick",
    `<div class="big">${{pct(m(adv,"answer_correctness"))}} <span style="font-size:14px;color:var(--muted)">correctness</span></div>
     <p>Rewrite + multi-query + HyDE + rerank + parent-doc composed together — but each stage adds latency and LLM calls, and the marginal gain over hybrid+rerank alone is smaller than the first upgrades. Composition helps; diminishing returns are real.</p>`, "#a371f7"),
  card("6.2 · GraphRAG", "Graph helps when questions are relational",
    `<div class="big">${{pct(byType(graph,"multi-hop","doc_recall"))}} <span style="font-size:14px;color:var(--muted)">multi-hop doc recall</span> <span class="delta up">${{deltaPct(graph)}} vs baseline</span></div>
     <p>Entity links catch the second hop that keyword/vector ranking misses. On single-hop questions the graph rarely changes the answer — it is a relational specialist, not a default upgrade.</p>`, "#3fb950"),
  card("6.3 · Agentic RAG", "Loops pay off only when they change the answer",
    `<div class="big">${{pct(byType(agentic,"multi-hop","doc_recall"))}} <span style="font-size:14px;color:var(--muted)">multi-hop doc recall</span> <span class="delta up">${{deltaPct(agentic)}} vs baseline</span></div>
     <p>Route → rewrite → retrieve → critique → re-retrieve. Mean ${{agentic.agent_stats ? agentic.agent_stats.mean_steps.toFixed(1) : "—"}} retrieval steps per question;
     ${{agentic.agent_stats ? agentic.agent_stats.n_no_retrieval : "—"}} questions routed past retrieval entirely. The loop earns its cost on multi-step questions — stop conditions are the hard part.</p>`, "#f778ba"),
  card("6.4 · Self-RAG", "Critique the draft, then decide",
    `<div class="big">${{pct(byType(selfrag,"tricky","faithfulness"))}} <span style="font-size:14px;color:var(--muted)">tricky-question faithfulness</span></div>
     <p>Generate → critique context support → re-retrieve once → answer "I don't know" rather than hallucinate. Declined ${{selfrag.agent_stats ? selfrag.agent_stats.n_declined : "—"}} questions.</p>`, "#ff7b72"),
].join("");

// ---------- self-rag panel ----------
const sp = selfrag.per_question;
const nDK = Object.keys(sp).filter(q => /don't know/i.test(sp[q].answer)).length;
document.getElementById("selfrag-panel").innerHTML = `
<div class="grid split">
  <div>
    <div class="big" style="margin:4px 0">${{nDK}} <span style="font-size:14px;color:var(--muted)">of ${{Object.keys(sp).length}} questions answered with "I don't know"</span></div>
    <p class="note">A pipeline that admits ignorance is more trustworthy than one that confabulates.
    The tricky questions in the golden set exist precisely to measure this behavior — a naive pipeline
    answers them confidently and wrongly; Self-RAG declines.</p>
  </div>
  <div><div class="chart-title">Faithfulness by question type (Self-RAG)</div><div id="chart-selfrag-types"></div></div>
</div>`;
hbar(document.getElementById("chart-selfrag-types"),
  ["single-hop","multi-hop","tricky","comparative"].map(t => ({{ name: t, value: byType(selfrag, t, "faithfulness") }})),
  x => x.value, {{ label: x => x.name, color: "#ff7b72" }});

// ---------- challenges ----------
const lat = R.challenges.latency;
document.getElementById("challenge-cards").innerHTML = [
  card("Bad retrieval", "Forcing worst-k chunks",
    `<div class="big">correctness ${{pct(bad.correctness_good_mean)}} → <span class="down">${{pct(bad.correctness_bad_mean)}}</span></div>
     <p class="note">${{bad.note}}</p>`),
  card("Latency", "Per-component timing",
    `<div class="big">${{lat.hybrid_plus_rerank_ms.toFixed(0)}} <span style="font-size:14px;color:var(--muted)">ms for hybrid+rerank</span></div>
     <p class="note">Everything except reranking and generation is single-digit ms. ${{lat.note}}</p>`),
  card("Cost", "Tokens as a design input",
    `<div class="big">${{meta.llm_offline ? "0" : R.variants.reduce((a,v)=>a+v.cost.usd,0).toFixed(3) + " $"}} <span style="font-size:14px;color:var(--muted)">total estimated spend</span></div>
     <p class="note">See the cost table above for per-variant token totals and how query transformations multiply input tokens.</p>`),
  card("Bad-retrieval detection", "Catch it before generation",
    `<div class="big">${{pct(mhBase)}} <span style="font-size:14px;color:var(--muted)">baseline multi-hop recall</span></div>
     <p class="note">Low similarity scores + missing keyword overlap flag bad retrieval <i>before</i> the LLM runs — the trigger for Self-RAG re-retrieval and "I don't know" answers.</p>`),
].join("");
const bi = R.challenges.boundary_integrity;
if (bi) {{
  document.getElementById("challenge-cards").insertAdjacentHTML("beforeend",
   card("Boundary integrity", "How much each chunker damages information",
    `<div class="big">${{(bi.fixed.fraction_intact*100).toFixed(0)}}% <span style="font-size:14px;color:var(--muted)">sentences intact, fixed-size</span></div>
     <p class="note">recursive ${{(bi.recursive.fraction_intact*100).toFixed(0)}}% · semantic ${{(bi.semantic.fraction_intact*100).toFixed(0)}}% · parent-child ${{(bi.parent_child.fraction_intact*100).toFixed(0)}}%.
     ${{bi.fixed.sentences_split}} corpus sentences get cut in half by fixed windows — an answer starting in one
     chunk and ending in the next is information no retriever can recover. ${{bi.note}}</p>`));
}}
const cd = R.challenges.chunk_damage;
if (cd) {{
  document.getElementById("challenge-cards").insertAdjacentHTML("beforeend",
   card("Chunk damage → answer quality", "The smoking gun",
    `<div class="big">${{pct(cd.correctness_damaged_subset.fixed)}} → <span class="up">${{pct(cd.correctness_damaged_subset.semantic)}}</span>
     <span style="font-size:14px;color:var(--muted)">fixed vs semantic correctness on the ${{cd.n_damaged_by_fixed}} questions fixed chunking damages</span></div>
     <p class="note">On the ${{cd.n_intact_under_fixed}} questions where fixed windows happen to keep the answer whole,
     correctness is ${{pct(cd.correctness_intact_subset.fixed)}} (fixed) vs ${{pct(cd.correctness_intact_subset.semantic)}} (semantic) — nearly tied.
     The damage is question-specific: ${{cd.note}}</p>`));
}}

// ---------- kb panel ----------
document.getElementById("kb-panel").innerHTML = `
<div class="grid split">
  <div>
    <h3>Deleting 07-reranking.md</h3>
    <div class="big">doc recall ${{pct(kb.doc_recall_affected_mean)}} <span style="font-size:14px;color:var(--muted)">on affected questions (${{kb.affected_question_ids.join(", ")}})</span></div>
    <p class="note">control questions: ${{pct(kb.doc_recall_control_mean)}} recall — unaffected.</p>
  </div>
  <div>
    <h3>Adding 16-token-budgets-and-prompt-templates.md</h3>
    <div class="big">q51 recall: ${{kb.q51_doc_recall_without_new_doc === null ? "—" : pct(kb.q51_doc_recall_without_new_doc)}} → ${{kb.q51_doc_recall_with_new_doc === null ? "—" : pct(kb.q51_doc_recall_with_new_doc)}}</div>
    <p class="note">${{kb.note}}</p>
  </div>
</div>`;

// ---------- full table ----------
const cols = ["doc_recall","hit_rate","mrr","ndcg","context_precision","faithfulness","answer_relevancy","answer_correctness"];
const colLabel = {{ doc_recall:"doc recall@5", hit_rate:"hit rate", mrr:"MRR", ndcg:"NDCG@5",
  context_precision:"ctx precision", faithfulness:"faithfulness", answer_relevancy:"relevancy", answer_correctness:"correctness" }};
let th = "<tr><th>variant</th><th>phase</th>" + cols.map(c => `<th>${{colLabel[c]}}</th>`).join("") + "</tr>";
let rows = sortedV().map(v => {{
  const bests = {{}};
  cols.forEach(c => {{
    const mx = Math.max(...V.map(x => m(x, c)));
    if (m(v,c) >= mx - 1e-9) bests[c] = true;
  }});
  return `<tr><td class="vname"><span class="phase-pill" style="background:${{phColor(v)}}"></span>${{v.name}}</td>
  <td style="color:var(--muted)">${{PHASE_LABELS[v.phase]}}</td>
  ${{cols.map(c => `<td class="${{bests[c] ? "win" : ""}}">${{(m(v,c)*100).toFixed(1)}}</td>`).join("")}}</tr>`;
}}).join("");
document.getElementById("full-table").innerHTML = th + rows;
document.getElementById("full-table").insertAdjacentHTML("afterend",
 meta.llm_offline
 ? `<p class="note">Faithfulness reads 100% across variants in offline mode by construction: the extractive
 generator answers with sentences lifted from the retrieved context, so every answer is trivially supported.
 It becomes discriminative with a live LLM — rerun with <code>--llm deepseek</code>/<code>--llm openai</code>
 (add <code>--judge</code> for LLM-judge scores).</p>`
 : `<p class="note">Faithfulness ${{meta.llm_judge ? "here is the LLM-judge score (claim-level support vs context)" :
 "here is the lexical support proxy (run with --judge for the LLM-judge version)"}}. The tricky-question
 breakdown below separates the variants, because declined answers ("I don't know") break the supported-claim pattern.</p>`);

// ---------- heatmap ----------
const qids = Object.keys(V[0].per_question);
const qMeta = Object.fromEntries((R.golden || []).map(i => [i.id, i]));
const W = 1180, cellW = 11, cellH = 13, padL = 200, padT = 60, padB = 10;
const H = padT + qids.length * cellH + padB;
let cells = "", head = "";
sortedV().forEach((v, vi) => {{
  const x = padL + vi * cellW;
  head += `<text transform="translate(${{x + cellW/2}}, 14) rotate(-55)" text-anchor="end"
    style="font-size:9px">${{v.name}}</text>`;
  qids.forEach((q, qi) => {{
    const val = v.per_question[q].answer_correctness ?? 0;
    const hue = Math.round(120 * Math.max(0, Math.min(1, val)));
    cells += `<rect x="${{x}}" y="${{padT + qi * cellH}}" width="${{cellW - 1}}" height="${{cellH - 1}}" rx="2"
      fill="hsl(${{hue}}, 65%, 45%)" opacity="0.9"><title>${{q}} — ${{v.name}}: ${{pct(val)}}</title></rect>`;
  }});
}});
qids.forEach((q, qi) => {{
  const meta2 = qMeta[q];
  const type = meta2 ? meta2.type : "";
  const label = (meta2 ? meta2.question : q).slice(0, 40);
  cells += `<text x="${{padL - 6}}" y="${{padT + qi * cellH + 9}}" text-anchor="end"
    style="font-size:9.5px">${{label}}${{label.length >= 40 ? "…" : ""}}</text>`;
}});
document.getElementById("heatmap").innerHTML =
  `<svg viewBox="0 0 ${{W}} ${{H}}">${{head}}${{cells}}</svg>
   <p class="note">Each cell = answer correctness for one question (row) under one variant (column).
   Red → low, green → high. Watch how weak variants go dark on whole rows (bad retrieval hits specific
   questions hard), and how the strongest columns — rerank, advanced, self-rag — lift the tricky rows.</p>`;

// ---------- methodology ----------
document.getElementById("method").innerHTML = `
<p><b>Corpus:</b> ${{meta.corpus_docs}} hand-written markdown docs about RAG techniques (${{meta.corpus_words.toLocaleString()}} words),
deliberately cross-referenced so multi-hop questions have ground truth.</p>
<p><b>Golden set:</b> ${{meta.golden_questions}} questions — ${{meta.golden_by_type["single-hop"]}} single-hop, ${{meta.golden_by_type["multi-hop"]}} multi-hop,
${{meta.golden_by_type["tricky"]}} tricky (unanswerable/misleading), ${{meta.golden_by_type["comparative"]}} comparative — each with a hand-written answer
and the docs that contain it. Every variant is scored against this same set.</p>
<p><b>Metrics:</b> retrieval — hit rate, precision/recall@k, MRR, NDCG, RAGAS-style context precision/recall
(both chunk- and document-level). Generation — faithfulness (lexical support proxy), answer relevancy,
answer correctness (token F1 vs golden); LLM-judge variants activate automatically with an API key.</p>
<p><b>Chunking:</b> fixed 60w/10w overlap · recursive (paragraph→sentence→word) · semantic (sentence-buffer,
90th-percentile break) · parent-child (40w children / 150w parents). Retrieval k = ${{meta.k}} everywhere.</p>
<p><b>Reproduce:</b> <code>pip install -r requirements.txt -r requirements-optional.txt</code> then
<code>python scripts/run_experiments.py</code> (add <code>--llm deepseek</code> or <code>--llm openai</code> with a key
in <code>.env</code>). This page is regenerated by <code>python site/build_report.py</code>.</p>
<p><b>${{meta.llm_offline ? "Honesty note (offline mode)" : "Live-LLM note"}}:</b> ${{meta.llm_offline ?
 "answers come from a deterministic extractive generator and query transforms are lexical heuristics — numbers here are a lower bound on what live LLMs achieve, and the report labels them as such." :
 "answers and query transforms were produced by a live LLM (" + meta.llm + ")" + (meta.llm_judge ? ", and the generation metrics include LLM-judge scores" : "") + ". Token costs are tracked per variant in the cost table."}}</p>`;

document.getElementById("footer").innerHTML =
  `Progressive RAG Documentation Assistant · <a href="https://github.com/isiomaC/progressive-rag">github.com/isiomaC/progressive-rag</a> ·
  built ${{meta.generated_at}} · embedder: ${{meta.embedder}} · llm: ${{meta.llm}} · ${{R.total_runtime_s}}s total runtime`;
</script>
</body>
</html>"""


if __name__ == "__main__":
    cfg = from_env()
    build_report(cfg)
