# Evaluation Metrics for RAG

## Two families of metrics

RAG quality has two independent halves, and you must measure both:

- **Retrieval metrics** ask "did we fetch the right chunks?" — computable
  without an LLM, against the golden set of relevant chunks.
- **Generation metrics** ask "is the final answer any good?" — needing a
  judge: either exact overlap with ground-truth answers or an LLM judge.

## Retrieval metrics

- **Hit rate** — fraction of questions where at least one relevant chunk is
  in the top-k.
- **Precision@k** — fraction of retrieved chunks that are relevant.
- **Recall@k** — fraction of all relevant chunks that were retrieved.
- **MRR (Mean Reciprocal Rank)** — mean of 1/rank of the first relevant
  chunk; rewards early hits.
- **NDCG@k** — graded, position-weighted relevance; the ranking-aware metric.
- **Context precision / recall** (RAGAS) — relevance of each retrieved chunk
  to the question, and coverage of the ground-truth context.

## Generation metrics (RAGAS triad)

- **Faithfulness** — is every claim in the answer supported by the retrieved
  context? Detects hallucination. The standard implementation is an LLM judge
  scoring each answer claim against the context.
- **Answer relevancy** — does the answer actually address the question? (An
  answer can be faithful and irrelevant.)
- **Answer correctness** — agreement with the ground-truth answer: overlap
  with the reference and, ideally, an LLM judge for semantic equivalence.
- **Context precision / recall** — the retrieval-side RAGAS pair.

## Offline proxies (used in demo mode)

Without an API key this project substitutes deterministic proxies:
faithfulness = fraction of answer sentences with lexical support in the
context; correctness = token-overlap F1 against the ground truth; relevancy
= lexical similarity between question and answer. These correlate with the
real metrics but are noisier — when an API key is present the same harness
switches to LLM judges automatically.

## LLM-as-judge caveats

Judges have biases: they prefer their own outputs, are swayed by position,
and differ between runs. Mitigations: fixed rubric prompts, multiple
evaluations, and keeping a deterministic lexical baseline like the one above
to sanity-check judge scores.

## The metric that changes pipeline design

Answer correctness is the metric every stakeholder asks for, but it is the
hardest to move with retrieval work alone: once the right chunks are in the
context, correctness is mostly the generator's business, and once the wrong
chunks are in, no generator can fix it. The pipeline-design consequence is
that retrieval work should be evaluated on *retrieval* metrics (recall,
MRR, NDCG) and judged good when the ceiling it enables is high. Treating
answer correctness as the scoreboard and retrieval metrics as the
diagnostics is the discipline that makes this project's comparisons
interpretable.

## Offline proxies, honestly labeled

The deterministic proxies used in offline mode deserve their caveats.
Faithfulness-as-coverage is one-directional by design — an answer sentence
counts as supported when most of its phrasing appears in the context —
which makes it an excellent hallucination detector for extractive answers
and a weak one for paraphrase-heavy answers. Correctness-as-F1 rewards
verbatim agreement with the golden answer and punishes equally-correct
rephrasings. Both correlate with their LLM-judge counterparts but compress
the scale; when an API key is present the harness switches to judges
automatically and the report labels which mode produced the numbers.
