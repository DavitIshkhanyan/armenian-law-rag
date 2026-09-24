# Evaluation Report — LLM Benchmark on the RAG Pipeline

**Question:** which LLM should the company standardise on for its Armenian/English legal Q&A assistant?

**Answer:** **Gemini 3.8 Flash.** It had the highest answer accuracy (88%), tied for best on Armenian questions
and was best on English ones. It made no unsupported claims, declined every out-of-scope question and had no
failed calls. Its latency was low and stable (95th percentile 2.7 s to the full answer). It is the most
expensive of the three per token, but the absolute cost is small: about $3.50–$7 per 1,000 questions. Its
Armenian output was also the cleanest. Details, trade-offs and caveats follow.

Final run: `eval/results/20260924-044849/` (raw answers in `raw.jsonl`, aggregates in `summary.json` and `summary.csv`).

## 1. Results

21 questions × 3 models, 63 answers, all judged. Each category has 7 questions. With 0 / 0.5 / 1 scoring, a
single half-point is 7.1 points within a category and 2.4 points overall.

| Metric | Gemini 3.8 Flash | gpt-oss-120b (Groq) | Nemotron 3 Super (OpenRouter) |
|---|---:|---:|---:|
| **Answer accuracy, all 21** | **88.1%** | 76.2% | 83.3% |
| Armenian questions (7) | **92.9%** | 85.7% | **92.9%** |
| English questions (7) | **92.9%** | 71.4% | 71.4% |
| Adversarial questions (7) | 78.6% | 71.4% | **85.7%** |
| **Citation F1** (answerable questions) | 94.3% | 90.2% | **96.1%** |
| Citation precision | 98.4% | 95.2% | **100%** |
| Citation recall | 92.2% | 88.2% | **94.1%** |
| **Hallucination rate** (unsupported claim) | **0%** | 9.5% (2 of 21) | **0%** |
| Out-of-scope questions declined | 3 of 3 | 3 of 3 | 3 of 3 (1 not in the fixed wording) |
| Answerable questions wrongly declined | 0 | 0 | 0 |
| **First token, median / p95** | 1.48 s / 2.64 s | **0.66 s / 1.11 s** | 4.48 s / 15.29 s |
| **Full answer, median / p95** | 1.57 s / 2.65 s | **0.90 s / 1.95 s** | 5.16 s / 17.77 s |
| Prompt tokens per question | 3,820 | **2,882** | 3,030 |
| Billed output tokens per question | 158 | 145 | 447 (mostly hidden reasoning) |
| **Cost per 1,000 questions, paid rates** | $3.46 ($6.92 from 2027) | $0.52 | **$0.44** |
| **Failed calls** (after retries) | **0** | **0** | **0** (2 retries after "upstream overloaded") |

Answer accuracy and hallucination are scored by the judge. Citation metrics, refusals, latency, tokens, cost and
failures are measured directly, without an LLM.

### By language

| | Gemini hy / en | gpt-oss hy / en | Nemotron hy / en |
|---|---|---|---|
| Accuracy (incl. adversarial in that language) | 94% / 83% | 78% / 75% | 94% / 75% |
| Prompt tokens | 5,671 / 2,432 | 3,489 / 2,427 | 3,818 / 2,440 |
| Median first token | 1.31 s / 1.54 s | 0.66 s / 0.62 s | 7.01 s / 3.94 s |

### Per question (1 = correct, 0.5 = partly correct, 0 = wrong; H = unsupported claim)

| | hy-01…07 | en-01…07 | adv-01…07 |
|---|---|---|---|
| Gemini | 1 1 1 **.5** 1 1 1 | 1 1 1 1 1 1 **.5** | 1 1 1 **.5** **.5** 1 **.5** |
| gpt-oss | 1 1 **.5** **.5H** 1 1 1 | 1 **.5** **.5** **.5** 1 1 **.5** | 1 1 1 **.5** 1 **0H** **.5** |
| Nemotron | 1 1 1 **.5** 1 1 1 | **.5** **.5** **.5** 1 1 1 **.5** | 1 1 1 **.5** 1 1 **.5** |

## 2. Recommendation

**Standardise on Gemini 3.8 Flash, with gpt-oss-120b as a fast, cheap fallback.**

Why Gemini:
- **Most accurate overall and on English**, and tied best on Armenian. Its errors were omissions of a
  secondary fact. It never got a legal rule backwards.
- **No unsupported claims**, correct refusals on all out-of-scope questions, and no wrong refusals.
- **Predictable latency.** The 95th percentile is under 3 s to the complete answer, with no outliers above
  8.5 s.
- **Best Armenian output.** Answers were fluent, cited with `[Հոդված N]` as instructed, and had no
  mixed-script text.
- **Most reliable free tier during the benchmark**: 0 errors and 0 retries.

Cost is not a deciding factor at this scale. A legal team asking 20,000 questions a month would spend about
$70/month now, or about $140/month at the post-2026 price. The per-token premium over the open-weight models
matters only at high volume.

**Why not the others as the default:**
- **gpt-oss-120b** is 2–4× faster and 7× cheaper, but it had the only real legal errors in the run:
  - adv-06: it told the user that reconsideration is *mandatory* before going to court, which reverses
    Art. 57(4);
  - hy-04: it presented cumulative licence-renewal conditions as alternatives.

  In a legal tool, one confidently inverted rule costs more than seconds of latency. It is a good fallback
  when Gemini is unavailable, and it is open-weights, so it can be self-hosted if data may not leave the
  company.
- **Nemotron 3 Super** was accurate on Armenian and adversarial questions and had the best citations. But on
  the free endpoint its latency was erratic (p95 15 s, maximum 20 s) and the quota is 50 requests/day. It also
  once produced a mixed-script word (see §3). It is worth re-testing on a dedicated paid endpoint.

**Before production:**
- use paid tiers, since free-tier limits made two of the three unusable for more than one user;
- review Google's data-processing terms for sending legal questions to an external API;
- re-run this benchmark when the model version changes.

## 3. Trade-offs

### Latency vs. accuracy

gpt-oss on Groq's hardware was fastest by a wide margin (first token 0.66 s) but was the least accurate.
Gemini was about 2× slower and 12 points more accurate. Both are fast enough for a research assistant, so
accuracy decides. Nemotron's latency is dominated by free-tier queueing and hidden reasoning: it produced
about 3× more output tokens than the others. Its slow tail comes from the service, not the model size.

**Reasoning settings mattered.** In a first run, Gemini used its default thinking budget. Its hidden
"thinking" tokens used up the output limit and **cut two answers off mid-sentence**. They also didn't appear in
`completion_tokens`, although they are billed. The final run uses `reasoning_effort=low` for all three models
and counts billed output as total minus prompt tokens. The first run was discarded (see §5).

### Armenian language handling

- **Accuracy:** all three models did well on the Armenian questions (86–93%). Retrieval found every required
  article for them (recall@5 = 100% for Armenian), and each prompt contains the authoritative Armenian text.
- **Token cost:** Gemini's tokenizer counts the same Armenian prompt as **5,671 tokens against about 3,500**
  for the other two (1.6×). English prompts are equal across all three (about 2,430). So Armenian costs Gemini
  proportionally more, and Armenian prompts get closer to per-minute token limits.
- **Following the citation format in Armenian:** gpt-oss used `[Article N]` instead of `[Հոդված N]` in 7 of
  its 9 Armenian answers. The parser accepts both, but a user sees a mixed-language answer.
- **Fluency glitches the judge did not penalise:** Nemotron wrote «գրтель» (Cyrillic letters inside an
  Armenian word, in hy-01). gpt-oss used «կոտորած» ("slaughtered") to describe a party affected by a decision
  (adv-06). Gemini had none. The judge scores legal content, so these only show up on manual review, which is
  one reason to keep a human spot-check (§5).

### Rate-limit reliability

The benchmark itself had no failures, because every model was paced to its free-tier limits and transient
errors were retried. Getting there showed how fragile free tiers are:

| Observation | Effect |
|---|---|
| Mistral free workspace: `x-ratelimit-limit-req-minute: 0` | Provider dropped |
| Gemini 2.5 Flash restricted to earlier users; Groq Llama 3.3 70B moved to enterprise | Lineup changed |
| OpenRouter free Gemma 4, GLM 5.2 and Qwen: "rate-limited upstream (shared pool)" | Not usable on demand |
| Groq free tier: 8K tokens/min | gpt-oss manages about **1 Armenian question per minute**, too little for more than one user |
| Groq free tier: 200K tokens/day (the judge model) | The judge hit the cap after about two full runs; the last 17 judgments were done the next day |
| OpenRouter free: 20 requests/min, **50 requests/day** | About two full benchmark runs per day |
| NVIDIA upstream "service temporarily overloaded" | 2 retries for Nemotron |
| Gemini free tier: 10 requests/min | No errors |

## 4. Error analysis

Across the 63 answers there were 20 half-scores and 1 zero. They fall into four groups:

1. **An omitted secondary fact (most of the half-scores).** For example: the right to repeal the decision if the
   99% fee is not paid (en-02); that the notice must state duration and reasons (en-01); the additional fine
   (hy-03); the fee on renewal (hy-04). These are prompt-addressable: ask for "all conditions and consequences".
2. **Legal logic errors, gpt-oss only.** "Or" instead of "and" (hy-04). "Approval *and* 45 days" instead of
   "approval *or* 45 days without objection" (en-03). The mandatory-reconsideration inversion (adv-06, scored 0
   and flagged as a hallucination).
3. **Retrieval limits every model equally.** en-07 needs Article 67 and adv-07 needs Article 55; neither was in
   the top 5. Every model correctly said it could not answer that part, and each question was capped at 0.5.
4. **Prompt design.** On the false-premise question (adv-04: "for how many years must operators keep call
   recordings?") all three models used the fixed refusal sentence. That is not wrong, but it doesn't correct the
   premise or mention the consent/court-order rule. The prompt should say "if the question assumes something
   the law does not say, say so" instead of forcing one refusal sentence.

Groups 3 and 4 are pipeline issues, not model issues, and they cost every model the same.

## 5. How the numbers were produced (and what was corrected)

**Setup.** Each model was called through the provider's OpenAI-compatible endpoint with temperature 0 and
`reasoning_effort=low`, streaming. Paid list prices are from September 2026 (per 1M tokens):

| Model | Input | Output |
|---|---:|---:|
| Gemini 3.8 Flash | $0.75 (introductory, $1.50 from 2027) | $3.75 ($7.50 from 2027) |
| gpt-oss-120b on Groq | $0.15 | $0.60 |
| Nemotron 3 Super, paid variant on OpenRouter | $0.08 | $0.45 |

The first plan was Gemini 2.5 Flash, Llama 3.3 70B and Mistral Small. None of them was usable on a free tier
(see §3).

**Protocol.**
- Retrieval ran **once per question**, and every model received the identical prompt.
- All models answered first, in parallel, each paced to its own limits. The judge scored the saved answers
  afterwards, so its rate limits never affected the models' timings.
- Rate-limit, timeout and overload errors were retried up to 3 times with backoff. Answers cut off by the
  output limit would count as `truncated` failures; there were none in the final run.

**Judge.** `qwen/qwen3.8-27b` on Groq, a model family that is not benchmarked, to avoid self-preference. It sees:
- the question and its type;
- the reference answer and key facts;
- the answer being graded;
- the retrieved articles relevant to it: those the answer cites first, then the expected ones.

It returns correctness (0 / 0.5 / 1) and whether any claim is unsupported *by the retrieved context*. Citation
accuracy, refusals, latency, tokens, cost and failures are computed without an LLM.

**Corrections made during evaluation.** Each was caught by manual review and applied equally to all models.

| Issue found | Fix |
|---|---|
| Gemini answers cut off by hidden thinking tokens; thinking tokens not counted | Same reasoning effort for all, output limit 4,096, billed output = total − prompt; **whole benchmark re-run** |
| Citation parser read `49(2)(1)` and `49(2)1` as articles 49, 2, 1 | Parser fixed with regression tests; citation metrics recomputed from saved answers |
| Judge context capped by characters, which cut cited articles and caused one false hallucination flag | Cited articles first, truncate instead of drop; affected question re-judged |
| Judge requests exceeded Groq's per-minute and per-day token limits | Pacing counts the full `max_tokens` reservation; failed judgments re-run with `--rescore --failed-only`, never re-rolling finished ones |

**Manual review.** I read every answer that scored below 1 (21 answers, which include both hallucination flags). The
judge's verdicts matched the reference answers and the law text. The judge was occasionally lenient: Nemotron
hy-02 got 1.0 while omitting "from publication". It did not penalise language-quality problems (§3).

## 6. Known limitations

- **Small evaluation set** (21 questions, 7 per category, one run). Gemini's lead over Nemotron (4.8 points) is one
  question's worth, which is suggestive but not statistically conclusive. The ranking of the bottom two is
  within noise.
- **A single LLM judge with a 3-point scale**, checked manually but not against a second judge or legal
  experts. The reference answers were written by an engineer reading the law, not by a lawyer.
- **Law version.** The Armenian source is the original 2005 text, and the English translation is consolidated
  to about 2009. The law has been amended many times since (the arlis.am history runs to 2025). The assistant
  answers about these texts, not the law currently in force.
- **Retrieval misses** cap some answers: Art. 55 and 61 were at rank 6–7 and Art. 67 below rank 10. There is no
  reranker.
- **Free-tier conditions.** Latency includes queueing on shared free infrastructure, especially for Nemotron,
  so paid dedicated endpoints would change the latency picture.
- **The prompt forces a fixed refusal sentence**, which makes out-of-scope detection deterministic but handles
  false-premise questions poorly.
- **Asymmetric sources.** Article 17.1 exists only in the English translation, so Armenian questions about it
  depend on cross-lingual retrieval.

## 7. With more time

1. **Build a larger answer key with the legal team:** about 100 questions, with reference answers written or
   reviewed by lawyers, including real questions from their work.
2. **Report confidence intervals:** several runs per model and a bootstrap over questions.
3. **Validate the judge:** a second judge from another family, and agreement with a human sample (Cohen's κ).
4. **Add a cross-encoder reranker** (e.g. bge-reranker-v2-m3) over the top 20 to recover the rank 6–7 misses.
5. **Ingest the current consolidated law** and store the edition date with each chunk, so answers can say
   "as amended on …".
6. **Refine the prompt:** premise correction instead of a fixed refusal; "list all conditions and
   consequences"; a stricter citation-label rule for Armenian answers.
7. **Add a citation verification step:** check each sentence against the text of the article it cites and flag
   unsupported sentences in the UI.
8. **Re-benchmark on paid endpoints**, including a self-hosted gpt-oss-120b or Nemotron, if data must stay
   in-house. Add more models, e.g. Gemini Flash-Lite as a cheaper tier.
9. **Collect production feedback:** a thumbs-up/down with comments in the UI, feeding new eval questions.
