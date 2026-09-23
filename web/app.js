const $ = (s) => document.querySelector(s);
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// Tabs
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab, .panel").forEach((el) => el.classList.remove("active"));
    t.classList.add("active");
    $("#" + t.dataset.tab).classList.add("active");
  })
);

// Read an SSE stream from a POST request, calling onEvent for every JSON event.
async function postStream(url, body, onEvent) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 2);
      if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
    }
  }
}

let MODELS = [];
async function loadModels() {
  MODELS = await (await fetch("/api/models")).json();
  $("#model").innerHTML = MODELS.map(
    (m) => `<option value="${m.key}" ${m.available ? "" : "disabled"}>${m.provider} — ${m.model}${m.available ? "" : " (no API key)"}</option>`
  ).join("");
  const firstAvailable = MODELS.find((m) => m.available);
  if (firstAvailable) $("#model").value = firstAvailable.key;
  document.dispatchEvent(new Event("models-loaded"));
}

// Wrap [Article N] / [Հոդված N] markers so they link to the retrieved source.
function renderAnswer(text, retrieved) {
  return esc(text).replace(/\[(Article|Art\.|Հոդված|Հոդ\.)\s*([^\]]+)\]/gi, (m, _w, inner) => {
    const nums = [...inner.matchAll(/(\d{1,2}(?:\.\d)?)/g)].map((x) => x[1]);
    const ok = nums.every((n) => retrieved.has(n));
    return `<span class="cite ${ok ? "" : "bad"}" data-art="${nums[0] || ""}" title="${ok ? "Show source" : "Cited article was not in the retrieved context"}">${m}</span>`;
  });
}

async function ask() {
  const question = $("#question").value.trim();
  if (!question) return;
  $("#askBtn").disabled = true;
  $("#answerBox").classList.remove("hidden");
  $("#answer").textContent = "…";
  $("#citations").innerHTML = $("#stats").innerHTML = $("#sources").innerHTML = $("#retrMeta").textContent = "";
  let text = "";
  const retrieved = new Set();
  try {
    await postStream("/api/ask", { question, model: $("#model").value, top_k: +$("#topk").value }, (ev) => {
      if (ev.type === "retrieval") {
        ev.articles.forEach((a) => retrieved.add(a.number));
        $("#retrMeta").textContent = `(${ev.lang === "hy" ? "Armenian" : "English"} question, ${ev.retrieval_s}s)`;
        $("#sources").innerHTML = ev.articles
          .map((a) => `<details class="source" id="src-${a.number}"><summary><b>${a.lang === "hy" ? "Հոդված" : "Article"} ${a.number}</b> — ${esc(a.title)}
            <span class="muted">score ${a.score}${a.partial ? " · excerpt" : ""}</span></summary><pre>${esc(a.text)}</pre></details>`)
          .join("");
      } else if (ev.type === "token") {
        text += ev.text;
        $("#answer").innerHTML = renderAnswer(text, retrieved);
      } else if (ev.type === "done") {
        const s = ev.stats;
        $("#answer").innerHTML = renderAnswer(ev.answer || "", retrieved);
        if (s.error) $("#answer").innerHTML += `<div class="error">Error: ${s.error} ${esc(s.error_detail || "")}</div>`;
        ev.citations.forEach((n) => $("#src-" + CSS.escape(n))?.classList.add("cited"));
        $("#citations").innerHTML = ev.refusal
          ? "Model reported the question is not covered by the law."
          : `Cited: ${ev.citations.join(", ") || "none"}` +
            (ev.uncited_context.length ? ` <span class="error">· not in retrieved context: ${ev.uncited_context.join(", ")}</span>` : "");
        const f = (x, d = 2) => (x == null ? "–" : (+x).toFixed(d));
        $("#stats").textContent = `TTFT ${f(s.ttft_s)}s · total ${f(s.total_s)}s · tokens ${s.prompt_tokens ?? "–"} in / ${s.completion_tokens ?? "–"} out` +
          ` · cost at paid rates $${f(ev.cost_usd, 5)}${s.retries ? ` · ${s.retries} retries` : ""}`;
      }
    });
  } catch (e) {
    $("#answer").innerHTML = `<span class="error">${esc(String(e))}</span>`;
  } finally {
    $("#askBtn").disabled = false;
  }
}

$("#askBtn").addEventListener("click", ask);
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask(); });
document.querySelectorAll(".ex").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); $("#question").value = a.textContent; }));
$("#answer").addEventListener("click", (e) => {
  const art = e.target.closest(".cite")?.dataset.art;
  const el = art && $("#src-" + CSS.escape(art));
  if (!el) return;
  el.open = true;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1200);
});

// ---------------- Benchmark tab ----------------
const METRICS = [
  ["answer_accuracy", "Answer accuracy (all)", "pct"],
  ["accuracy_hy", "  · Armenian questions", "pct"],
  ["accuracy_en", "  · English questions", "pct"],
  ["accuracy_adversarial", "  · Adversarial", "pct"],
  ["citation_f1", "Citation F1 (answerable)", "pct"],
  ["citation_precision", "Citation precision", "pct"],
  ["citation_recall", "Citation recall", "pct"],
  ["hallucination_rate", "Hallucination rate", "pct"],
  ["refusal_on_out_of_scope", "Refuses out-of-scope", "pct"],
  ["false_refusal_on_answerable", "False refusals", "pct"],
  ["ttft_p50_s", "TTFT p50 (s)", "num"],
  ["ttft_p95_s", "TTFT p95 (s)", "num"],
  ["total_p50_s", "Total time p50 (s)", "num"],
  ["total_p95_s", "Total time p95 (s)", "num"],
  ["prompt_tokens_avg", "Prompt tokens (avg)", "int"],
  ["completion_tokens_avg", "Completion tokens (avg)", "int"],
  ["cost_usd_per_1k_questions", "Cost / 1k questions (USD, paid rates)", "usd"],
  ["cost_usd_total", "Cost of this run (USD)", "usd5"],
  ["failure_rate", "Failure rate", "pct"],
  ["failures", "Failures by type", "obj"],
  ["retries", "Retries", "int"],
  ["judge_errors", "Judge errors", "int"],
];
function fmt(v, kind) {
  if (v == null) return "–";
  if (kind === "pct") return (100 * v).toFixed(1) + "%";
  if (kind === "num") return (+v).toFixed(2);
  if (kind === "int") return Math.round(v).toLocaleString();
  if (kind === "usd") return "$" + (+v).toFixed(3);
  if (kind === "usd5") return "$" + (+v).toFixed(5);
  if (kind === "obj") return Object.entries(v).map(([k, n]) => `${k}: ${n}`).join(", ") || "none";
  return v;
}
function renderSummary(summary) {
  if (!summary.length) { $("#benchSummary").textContent = "No results."; return; }
  const head = `<tr><th>Metric</th>${summary.map((s) => `<th>${s.model}<br><span class="muted">${esc(s.provider_model)}</span></th>`).join("")}</tr>`;
  const body = METRICS.map(([k, label, kind]) =>
    `<tr><td>${label.replace(/^  /, "&nbsp;&nbsp;")}</td>${summary.map((s) => `<td class="num">${fmt(s[k], kind)}</td>`).join("")}</tr>`).join("");
  $("#benchSummary").innerHTML = `<table>${head}${body}</table>`;
}
function renderRows(rows) {
  const r = [...rows].sort((a, b) => a.qid.localeCompare(b.qid) || a.model.localeCompare(b.model));
  $("#benchRows").innerHTML = `<table><tr><th>Q</th><th>Model</th><th>Score</th><th>Halluc.</th><th>Cited</th><th>Retrieved</th><th>TTFT</th><th>Answer</th></tr>` +
    r.map((x) => `<tr><td>${x.qid}</td><td>${x.model}</td>
      <td class="num ${x.correctness === 1 ? "ok" : x.correctness === 0 ? "fail" : ""}">${x.error ? `<span class="fail">${x.error}</span>` : fmt(x.correctness)}</td>
      <td>${x.hallucination == null ? "–" : x.hallucination ? '<span class="fail">yes</span>' : "no"}</td>
      <td>${x.citations.join(", ")}</td><td class="muted">${x.retrieved_articles.join(", ")}</td>
      <td class="num">${fmt(x.ttft_s, "num")}</td>
      <td><details><summary>${esc((x.answer || "").slice(0, 60))}…</summary><pre>${esc(x.answer || "")}</pre>
        ${x.judge_rationale ? `<p class="muted">Judge: ${esc(x.judge_rationale)}</p>` : ""}</details></td></tr>`).join("") + "</table>";
}
async function loadRuns(select) {
  const runs = await (await fetch("/api/benchmark/runs")).json();
  $("#runs").innerHTML = `<option value="">—</option>` + runs.map((r) => `<option>${r}</option>`).join("");
  if (select && runs.includes(select)) $("#runs").value = select;
  else if (!select && runs.length) $("#runs").value = runs[0];
  if ($("#runs").value) showRun($("#runs").value);
}
async function showRun(name) {
  const data = await (await fetch("/api/benchmark/runs/" + encodeURIComponent(name))).json();
  $("#runMeta").textContent = `(run ${name}, judge: ${data.meta.judge_model || "?"}, ${data.rows.length} answers)`;
  renderSummary(data.summary);
  renderRows(data.rows);
}
$("#runs").addEventListener("change", (e) => e.target.value && showRun(e.target.value));

async function runBenchmark() {
  const models = [...document.querySelectorAll(".benchModel:checked")].map((c) => c.value);
  if (!models.length) return;
  const limit = +$("#benchLimit").value || null;
  $("#benchBtn").disabled = true;
  const cells = {};
  let t0 = Date.now(), outDir = "";
  try {
    await postStream("/api/benchmark", { models, limit }, (ev) => {
      if (ev.type === "start") {
        outDir = ev.out_dir.split("/").pop();
        $("#benchStatus").textContent = `Retrieving context for ${ev.n_questions} questions…`;
        $("#benchProgress").innerHTML = models.map((m) => `<div><b>${m}</b> <span id="prog-${m}"></span></div>`).join("");
      } else if (ev.type === "retrieval_done") {
        $("#benchStatus").textContent = "Querying models (paced to free-tier limits)…";
      } else if (ev.type === "progress") {
        cells[ev.model] = (cells[ev.model] || 0) + 1;
        const mark = ev.error ? `<span class="fail" title="${ev.error}">✗</span>` : ev.correctness === 1 ? '<span class="ok">●</span>' : ev.correctness === 0 ? '<span class="fail">●</span>' : "◐";
        document.getElementById("prog-" + ev.model).innerHTML += mark;
        $("#benchStatus").textContent = `Running… ${Math.round((Date.now() - t0) / 1000)}s`;
      } else if (ev.type === "summary") {
        $("#benchStatus").textContent = `Done in ${Math.round((Date.now() - t0) / 1000)}s`;
        loadRuns(outDir);
      }
    });
  } catch (e) {
    $("#benchStatus").innerHTML = `<span class="error">${esc(String(e))}</span>`;
  } finally {
    $("#benchBtn").disabled = false;
  }
}
$("#benchBtn").addEventListener("click", runBenchmark);
document.addEventListener("models-loaded", () => {
  $("#benchModels").innerHTML = MODELS.map((m) =>
    `<label><input type="checkbox" class="benchModel" value="${m.key}" ${m.available ? "checked" : "disabled"}> ${m.provider} ${esc(m.model)}</label>`).join(" ");
});

loadModels();
loadRuns();
