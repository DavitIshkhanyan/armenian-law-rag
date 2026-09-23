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

loadModels();
