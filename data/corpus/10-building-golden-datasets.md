# Building a Golden Evaluation Dataset

## What a golden set is

A golden set is a fixed collection of questions, each annotated with a
ground-truth answer and the chunks (or documents) that contain it. Every
experiment in this project is scored against this one set, which is what
makes comparisons between variants meaningful: same questions, same
reference answers, different pipelines.

## How to build one

- **Size.** 50–100 questions is enough to rank techniques reliably; smaller
  sets produce noisy rankings.
- **Mix.** Single-hop questions (answer in one chunk), multi-hop questions
  (need two or more documents), tricky questions (ambiguous, negative, or
  unanswerable from the corpus). Our set: 31 single-hop, 8 multi-hop, 6
  tricky, 6 comparative.
- **Ground truth.** Write the ideal answer by hand from the corpus, and
  resolve which chunks contain it — in this project a build script matches
  answer sentences back to chunks automatically.
- **Provenance.** Every question records which documents it targets, so
  failures can be analyzed per-document and per-type.

## Pitfalls

- **Contamination.** If the same person writes the corpus and the questions,
  questions drift towards the corpus vocabulary — which flatters lexical
  retrievers. Writing questions from the *reader's* perspective helps.
- **Unanswerable questions are missing.** Without them, a pipeline that
  always answers confidently looks perfect. Tricky questions measure the
  "I don't know" behavior that Self-RAG and corrective RAG add.
- **Stale references.** When the knowledge base changes, golden answers can
  become wrong. The knowledge-base update experiment rebuilds a subset of
  the corpus and checks which golden entries break.

## Bad-retrieval detection

Low-quality retrievals are detectable before the LLM runs: very low maximum
similarity, low rerank scores, or no chunk sharing keywords with the
question. Flagging these lets the pipeline either re-retrieve, answer "I
don't know", or log the case for dataset fixes. The Phase 5 challenge lab
forces bad retrieval deliberately and measures how much answer quality
degrades, to calibrate what "bad" looks like numerically.

## Anatomy of a good golden entry

A golden entry is four things: a question written from the reader's
perspective, a hand-written ideal answer, the list of documents that
contain the answer, and the concrete chunk ids resolved against a specific
chunking strategy (because chunk boundaries move between strategies, the
resolution is re-derived per strategy — see the architecture document).
The question types are deliberately mixed so that a pipeline cannot game
one style:

- single-hop questions test whether one retrieval pass finds one answer,
- multi-hop questions test whether the context can contain two documents
  at once,
- tricky questions test honesty — some have no answer in the corpus, some
  are misleading, some mix two unrelated techniques,
- comparative questions test whether the answer synthesizes rather than
  echoes.

A pipeline that scores well on all four types is a pipeline that retrieves,
synthesizes, and knows when to stop.
