const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const isHy = (s) => /[Ա-֏]/.test(s);

// ---------------- Tabs ----------------
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((el) => { el.classList.remove("active"); el.setAttribute("aria-selected", "false"); });
    document.querySelectorAll(".panel").forEach((el) => el.classList.remove("active"));
    t.classList.add("active");
    t.setAttribute("aria-selected", "true");
    $("#" + t.dataset.tab).classList.add("active");
  })
);

// Read an SSE stream from a POST request, calling onEvent for every JSON event.
async function postStream(url, body, onEvent) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) {
    let detail = await res.text();
    try { detail = JSON.parse(detail).detail; detail = typeof detail === "string" ? detail : JSON.stringify(detail); } catch {}
    throw new Error(`The server rejected the request: ${detail}`);
  }
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

const shortModel = (m) => m.replace(/:free$/, "");
let MODELS = [];
async function loadModels() {
  MODELS = await (await fetch("/api/models")).json();
  $("#model").innerHTML = MODELS.map(
    (m) => `<option value="${m.key}" ${m.available ? "" : "disabled"}>${esc(m.provider)}: ${esc(shortModel(m.model))}${m.available ? "" : " (no API key)"}</option>`
  ).join("");
  const first = MODELS.find((m) => m.available);
  if (first) $("#model").value = first.key;
  $("#askBtn").disabled = !first;
  if (!first) $("#answerArea").insertAdjacentHTML("afterbegin",
    `<div class="error-box">No model has an API key. Add keys to .env and restart the server.</div>`);
  $("#benchModels").innerHTML = MODELS.map((m) =>
    `<label><input type="checkbox" class="benchModel" value="${m.key}" ${m.available ? "checked" : "disabled"}> ${esc(m.provider)}: ${esc(shortModel(m.model))}</label>`).join("");
}

// ---------------- Ask ----------------
const artLabel = (lang, n) => `${lang === "hy" ? "Հոդված" : "Article"} ${n}`;

// Escape, keep **bold**, and turn [Article N] / [Հոդված N] markers into buttons linked to the law pane.
function renderAnswer(text, retrieved) {
  return esc(text.replace(/[\u00A0\u202F\u2009]/g, " "))
    .replace(/^[ \t]*[*-][ \t]+/gm, "• ")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\[(Article|Art\.|Հոդված|Հոդ\.)\s*([^\]]+)\]/gi, (m, _w, inner) => {
      const nums = [...inner.replace(/\([^)]*\)/g, " ").matchAll(/\d{1,2}(?:\.\d)?/g)].map((x) => x[0]);
      const ok = nums.length && nums.every((n) => retrieved.has(n));
      return `<button type="button" class="cite ${ok ? "" : "bad"}" data-art="${nums[0] || ""}"
        title="${ok ? "Show this article" : "This article was not among the retrieved articles"}">${m.slice(1, -1)}</button>`;
    });
}

function renderSources(ev) {
  if (!ev.articles.length) { $("#sources").innerHTML = `<p class="law-empty">No articles were retrieved.</p>`; return; }
  $("#sources").innerHTML = ev.articles.map((a) => `
    <section class="statute" id="src-${esc(a.number)}" lang="${a.lang}">
      <div class="statute-num" aria-label="${a.lang === "hy" ? "Հոդված" : "Article"} ${esc(a.number)}">${esc(a.number)}</div>
      <div class="statute-main">
        <div class="statute-head" data-toggle>
          <span class="statute-title">${esc(a.title)}</span>
          <span class="statute-tags">${[
            `<span class="tag-cited" hidden>Cited in the answer</span>`,
            a.lang !== ev.lang ? (a.lang === "en" ? "English text only" : "Armenian text only") : "",
            a.partial ? "Excerpt" : "",
          ].filter(Boolean).join(" ")}</span>
        </div>
        <div class="statute-body">${esc(a.text)}</div>
        <button type="button" class="statute-toggle" data-toggle>Show full text</button>
      </div>
    </section>`).join("");
}

function setOpen(el, open) {
  el.classList.toggle("open", open);
  el.querySelector(".statute-toggle").textContent = open ? "Show less" : "Show full text";
}

$("#sources").addEventListener("click", (e) => {
  if (!e.target.closest("[data-toggle]")) return;
  const el = e.target.closest(".statute");
  setOpen(el, !el.classList.contains("open"));
});

function showArticle(num) {
  const el = num && document.getElementById("src-" + num);
  if (!el) return;
  setOpen(el, true);
  const behavior = matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  const pane = $("#sources");
  if (getComputedStyle(pane.parentElement).position === "sticky") {
    // Desktop: scroll inside the law pane only, so the question and answer stay in view.
    pane.scrollTo({ top: el.offsetTop, behavior });
  } else {
    el.scrollIntoView({ behavior, block: "start" });
  }
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1400);
}
$("#answer").addEventListener("click", (e) => showArticle(e.target.closest(".cite")?.dataset.art));

const fmtS = (x) => (x == null ? "n/a" : `${(+x).toFixed(2)} s`);

async function ask() {
  const question = $("#question").value.trim();
  if (!question) { $("#question").focus(); return; }
  if (!$("#model").value || $("#askBtn").disabled) return;
  const qLang = isHy(question) ? "hy" : "en";
  $("#askBtn").disabled = true;
  $("#askBtn").textContent = "Answering…";
  $("#answerArea").classList.remove("empty");
  $("#answerCard").hidden = false;
  $("#answerCard").classList.remove("refusal");
  $("#answer").innerHTML = "";
  $("#answer").classList.add("streaming");
  $("#answer").lang = qLang;
  $("#answerStatus").textContent = "Finding the relevant articles…";
  $("#citations").innerHTML = $("#stats").textContent = "";
  $("#sources").innerHTML = "";
  let text = "";
  const retrieved = new Set();
  try {
    await postStream("/api/ask", { question, model: $("#model").value, top_k: +$("#topk").value }, (ev) => {
      if (ev.type === "retrieval") {
        ev.articles.forEach((a) => retrieved.add(a.number));
        renderSources(ev);
        $("#retrMeta").textContent = `${ev.articles.length} articles retrieved for this ${ev.lang === "hy" ? "Armenian" : "English"} question in ${ev.retrieval_s.toFixed(2)} s. Articles the answer cites are marked in blue.`;
        $("#answerStatus").textContent = "Writing the answer…";
      } else if (ev.type === "token") {
        text += ev.text;
        $("#answer").innerHTML = renderAnswer(text, retrieved);
      } else if (ev.type === "done") {
        const s = ev.stats;
        $("#answer").classList.remove("streaming");
        $("#answer").innerHTML = renderAnswer(ev.answer || "", retrieved);
        const model = MODELS.find((m) => m.key === $("#model").value);
        $("#answerStatus").textContent = `Answer from ${model ? shortModel(model.model) : s.model}`;
        if (s.error) {
          $("#answer").insertAdjacentHTML("beforeend",
            `<div class="error-box">The model call failed (${esc(s.error)}). ${esc(s.error_detail || "")} Try again or choose another model.</div>`);
        }
        ev.citations.forEach((n) => {
          const el = document.getElementById("src-" + n);
          if (el) { el.classList.add("cited"); el.querySelector(".tag-cited").hidden = false; }
        });
        if (ev.refusal) {
          $("#answerCard").classList.add("refusal");
          $("#citations").textContent = "The retrieved articles do not cover this question, so the assistant did not answer it.";
        } else if (ev.citations.length) {
          $("#citations").innerHTML = `Cites ${ev.citations.map((n) => artLabel(qLang, n)).join(", ")}.` +
            (ev.uncited_context.length ? ` <span class="warn">${ev.uncited_context.map((n) => artLabel(qLang, n)).join(", ")} ${ev.uncited_context.length > 1 ? "were" : "was"} not among the retrieved articles.</span>` : "");
        } else if (!s.error) {
          $("#citations").innerHTML = `<span class="warn">The answer has no citations.</span>`;
        }
        if (!s.error) {
          const cost = ev.cost_usd == null ? "" : `, $${ev.cost_usd.toFixed(5)} at paid rates`;
          $("#stats").textContent = `First token ${fmtS(s.ttft_s)}, full answer ${fmtS(s.total_s)}. ` +
            `${(s.prompt_tokens ?? 0).toLocaleString()} prompt and ${(s.completion_tokens ?? 0).toLocaleString()} answer tokens${cost}.` +
            (s.retries ? ` Retried ${s.retries} time${s.retries > 1 ? "s" : ""}.` : "");
        }
      }
    });
  } catch (e) {
    $("#answer").classList.remove("streaming");
    $("#answerStatus").textContent = "";
    $("#answer").innerHTML = `<div class="error-box">${esc(e.message)}</div>`;
  } finally {
    $("#askBtn").disabled = false;
    $("#askBtn").textContent = "Ask";
  }
}

$("#askForm").addEventListener("submit", (e) => { e.preventDefault(); ask(); });
$("#question").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); ask(); } });
document.querySelectorAll(".ex").forEach((b) => b.addEventListener("click", () => { $("#question").value = b.textContent.trim(); ask(); }));

// ---------------- Benchmark ----------------
// [key, label, format, better ("high" | "low" | null), indented]; a single-item row is a group heading.
const METRICS = [
  ["Quality"],
  ["answer_accuracy", "Answer accuracy", "pct", "high"],
  ["accuracy_hy", "Armenian questions", "pct", "high", true],
  ["accuracy_en", "English questions", "pct", "high", true],
  ["accuracy_adversarial", "Adversarial questions", "pct", "high", true],
  ["citation_f1", "Citation F1", "pct", "high"],
  ["citation_precision", "Citation precision", "pct", "high", true],
  ["citation_recall", "Citation recall", "pct", "high", true],
  ["hallucination_rate", "Hallucination rate", "pct", "low"],
  ["refusal_on_out_of_scope", "Declines out-of-scope questions", "pct", "high"],
  ["false_refusal_on_answerable", "Declines answerable questions", "pct", "low"],
  ["Speed"],
  ["ttft_p50_s", "First token, median", "sec", "low"],
  ["ttft_p95_s", "First token, 95th percentile", "sec", "low"],
  ["total_p50_s", "Full answer, median", "sec", "low"],
  ["total_p95_s", "Full answer, 95th percentile", "sec", "low"],
  ["Tokens and cost"],
  ["prompt_tokens_avg", "Prompt tokens per question", "int", "low"],
  ["completion_tokens_avg", "Billed output tokens per question", "int", null],
  ["reasoning_tokens_avg", "of which hidden reasoning", "int", null, true],
  ["cost_usd_per_1k_questions", "Cost per 1,000 questions at paid rates", "usd", "low"],
  ["cost_usd_total", "Cost of this run", "usd5", "low"],
  ["Reliability"],
  ["failure_rate", "Failed calls", "pct", "low"],
  ["failures", "Failures by type", "obj", null],
  ["retries", "Retries", "int", "low"],
  ["judge_errors", "Judge errors", "int", null],
];

function fmt(v, kind) {
  if (v == null) return "n/a";
  if (kind === "pct") return (100 * v).toFixed(1) + "%";
  if (kind === "sec") return (+v).toFixed(2) + " s";
  if (kind === "int") return Math.round(v).toLocaleString();
  if (kind === "usd") return "$" + (+v).toFixed(3);
  if (kind === "usd5") return "$" + (+v).toFixed(5);
  if (kind === "obj") return Object.entries(v).map(([k, n]) => `${k} ${n}`).join(", ") || "none";
  return v;
}

function renderSummary(summary) {
  if (!summary.length) { $("#benchSummary").innerHTML = `<p class="muted">This run has no results.</p>`; return; }
  const head = `<thead><tr><th>Metric</th>${summary.map((s) => `<th>${esc(shortModel(s.provider_model))}</th>`).join("")}</tr></thead>`;
  const rows = METRICS.map(([key, label, kind, better, sub]) => {
    if (!label) return `<tr class="group"><td colspan="${summary.length + 1}">${key}</td></tr>`;
    const vals = summary.map((s) => s[key]);
    const nums = vals.filter((v) => typeof v === "number");
    const best = better && nums.length > 1 && new Set(nums).size > 1 ? (better === "high" ? Math.max(...nums) : Math.min(...nums)) : null;
    return `<tr><td class="${sub ? "sub" : ""}">${label}</td>${vals.map((v) =>
      `<td class="num ${best != null && v === best ? "best" : ""}">${fmt(v, kind)}</td>`).join("")}</tr>`;
  }).join("");
  $("#benchSummary").innerHTML = `<table class="data">${head}<tbody>${rows}</tbody></table>
    <p class="muted">The best value in each row is shown in green.</p>`;
}

const scoreClass = (x) => (x.error ? "err" : x.correctness === 1 ? "s1" : x.correctness === 0.5 ? "s05" : x.correctness === 0 ? "s0" : x.answered ? "answered" : "");
const GROUPS = [["answerable_hy", "Armenian"], ["answerable_en", "English"], ["adversarial", "Adversarial"]];

// Model x question grid; `cells` maps "model|qid" to a result row or progress event.
function renderGrid(models, questions, cells) {
  const byGroup = GROUPS.map(([cat, name]) => [name, questions.filter((q) => q.category === cat)]).filter(([, qs]) => qs.length);
  const groupHead = `<tr><th></th>${byGroup.map(([name, qs]) => `<th class="group" colspan="${qs.length}">${name}</th>`).join("")}</tr>`;
  const qHead = `<tr><th></th>${byGroup.flatMap(([, qs]) => qs.map((q) => `<th title="${esc(q.question)}">${esc(q.id.split("-")[1])}</th>`)).join("")}</tr>`;
  const body = models.map((m) => `<tr><th class="model">${esc(m)}</th>${byGroup.flatMap(([, qs]) => qs.map((q) => {
    const c = cells[`${m}|${q.id}`];
    const title = !c ? `${q.id}: waiting` : c.error ? `${q.id}: call failed (${c.error})`
      : c.correctness != null ? `${q.id}: score ${c.correctness}` : c.answered ? `${q.id}: answered, waiting for the judge` : `${q.id}: not judged`;
    return `<td><div class="cell ${c ? scoreClass(c) : ""}" title="${esc(title)}"></div></td>`;
  })).join("")}</tr>`).join("");
  $("#grid").innerHTML = `<table class="qgrid">${groupHead}${qHead}${body}</table>
    <div class="legend"><span><i class="cell s1"></i>Correct</span><span><i class="cell s05"></i>Partly correct</span>
    <span><i class="cell s0"></i>Wrong</span><span><i class="cell err"></i>Call failed</span>
    <span><i class="cell answered"></i>Answered, not yet scored</span><span><i class="cell"></i>Waiting</span></div>`;
}

function renderRows(rows) {
  const r = [...rows].sort((a, b) => a.qid.localeCompare(b.qid) || a.model.localeCompare(b.model));
  $("#benchRows").innerHTML = `<table class="data"><thead><tr><th>Question</th><th>Model</th><th>Score</th><th>Unsupported claims</th>
    <th>Cited</th><th>Retrieved</th><th>First token</th><th>Answer</th></tr></thead><tbody>` +
    r.map((x) => `<tr>
      <td title="${esc(x.question)}">${esc(x.qid)}</td>
      <td>${esc(x.model)}</td>
      <td class="num"><span class="score ${scoreClass(x)}">${x.error ? esc(x.error) : fmt(x.correctness)}</span></td>
      <td>${x.hallucination == null ? "n/a" : x.hallucination ? `<span class="score s0">Yes</span>` : "No"}</td>
      <td>${esc(x.citations.join(", ")) || "none"}</td>
      <td class="muted">${esc(x.retrieved_articles.join(", "))}</td>
      <td class="num">${x.ttft_s == null ? "n/a" : (+x.ttft_s).toFixed(2) + " s"}</td>
      <td class="ans-cell"><details><summary>Read</summary><pre lang="${x.lang}">${esc(x.answer || "")}</pre>
        ${x.judge_rationale ? `<p class="muted">Judge: ${esc(x.judge_rationale)}</p>` : ""}</details></td></tr>`).join("") +
    "</tbody></table>";
}

let QUESTIONS = [];
async function loadRuns(select) {
  const runs = await (await fetch("/api/benchmark/runs")).json();
  $("#runs").innerHTML = runs.length ? runs.map((r) => `<option>${esc(r)}</option>`).join("") : `<option value="">None yet</option>`;
  const pick = select && runs.includes(select) ? select : runs[0];
  if (pick) { $("#runs").value = pick; showRun(pick); }
}

async function showRun(name) {
  const data = await (await fetch("/api/benchmark/runs/" + encodeURIComponent(name))).json();
  const models = [...new Set(data.rows.map((r) => r.model))];
  const cells = Object.fromEntries(data.rows.map((r) => [`${r.model}|${r.qid}`, r]));
  const inRun = new Set(data.rows.map((r) => r.qid));
  renderGrid(models, QUESTIONS.filter((q) => inRun.has(q.id)), cells);
  $("#runMeta").textContent = `for run ${name}, judged by ${data.meta.judge_model || "unknown"}`;
  renderSummary(data.summary);
  renderRows(data.rows);
}
$("#runs").addEventListener("change", (e) => e.target.value && showRun(e.target.value));

async function runBenchmark() {
  const models = [...document.querySelectorAll(".benchModel:checked")].map((c) => c.value);
  if (!models.length) { $("#benchStatus").textContent = "Choose at least one model."; return; }
  const limit = +$("#benchLimit").value || null;
  const qs = limit ? QUESTIONS.slice(0, limit) : QUESTIONS;
  $("#benchBtn").disabled = true;
  const cells = {};
  const t0 = Date.now();
  let outDir = "", answered = 0, scored = 0;
  const total = models.length * qs.length;
  const tick = setInterval(() => {
    const secs = Math.round((Date.now() - t0) / 1000);
    $("#benchStatus").textContent = answered < total
      ? `Models have answered ${answered} of ${total} (${secs} s). Requests are paced to each provider's free-tier limits.`
      : `All answers are in. The judge has scored ${scored} of ${total} (${secs} s).`;
  }, 1000);
  try {
    await postStream("/api/benchmark", { models, limit }, (ev) => {
      if (ev.type === "start") {
        outDir = ev.out_dir.split("/").pop();
        renderGrid(models, qs, cells);
      } else if (ev.type === "answered") {
        answered++;
        cells[`${ev.model}|${ev.qid}`] = { ...ev, answered: true };
        renderGrid(models, qs, cells);
      } else if (ev.type === "progress") {
        scored++;
        cells[`${ev.model}|${ev.qid}`] = ev;
        renderGrid(models, qs, cells);
      } else if (ev.type === "summary") {
        clearInterval(tick);
        $("#benchStatus").textContent = `Finished in ${Math.round((Date.now() - t0) / 1000)} s. Saved as run ${outDir}.`;
        loadRuns(outDir);
      }
    });
  } catch (e) {
    $("#benchStatus").textContent = e.message;
  } finally {
    clearInterval(tick);
    $("#benchBtn").disabled = false;
  }
}
$("#benchBtn").addEventListener("click", runBenchmark);

(async () => {
  try { QUESTIONS = await (await fetch("/api/questions")).json(); } catch { QUESTIONS = []; }
  await loadModels();
  loadRuns();
})();
