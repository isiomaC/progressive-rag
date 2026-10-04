# Query Transformation

## Why transform the query

Users write short, ambiguous, or poorly worded questions. The retriever is a
pattern matcher, so retrieval quality is capped by the quality of the query
text. Query transformation rewrites or expands the query before retrieval —
at the price of extra LLM calls and latency.

## Multi-query retrieval

Ask the LLM to produce several reformulations of the question (we use three:
the original plus two variants), retrieve for each, and merge the result
lists with RRF. If one formulation matches the corpus vocabulary and another
does not, the merge covers both. Multi-query shines on ambiguous questions
and costs N retrieval passes plus one LLM call.

## Query rewriting

A single LLM pass rewrites the raw question into a self-contained, concrete
retrieval query: expanding pronouns, adding context, and stating the
information need explicitly. Rewriting is cheaper than multi-query (one
extra LLM call instead of one call plus multiple retrievals) and is the
default first transformation in production pipelines.

## HyDE — Hypothetical Document Embeddings

HyDE inverts the pipeline: ask the LLM to *write a short answer* to the
question, embed that hypothetical answer, and retrieve chunks similar to it.
It works because a hypothetical answer is closer in vocabulary to real
documents than the raw question is — the model guesses how the corpus would
phrase the answer. HyDE costs one LLM call per query and can hurt on
questions where the model's guess is confidently wrong, so it is best kept
behind a routing decision rather than applied universally.

## Step-back prompting

For questions that ask about a specific instance of a general concept, ask a
broader "step-back" question first ("what is RAG?" before "why is my RAG
pipeline slow?"), retrieve on the broad question, then answer the specific
one with both retrievals. Step-back helps when the corpus explains the
concept but not the instance.

## Cost accounting

Every transformation is an LLM call before retrieval even starts. On a
budget, the order to adopt is: rewrite (cheapest), then multi-query, then
HyDE (riskiest). The latency and cost experiment in this project measures
each technique's added milliseconds and tokens.

## Why transformations fail

Each transformation has a characteristic failure mode, and knowing them
makes the experiments readable:

- **Rewrite** fails by dropping the essential term — "why does the
  reranker document mention cross-encoders?" rewritten as "cross-encoders"
  loses the provenance constraint the question actually carried.
- **Multi-query** fails by producing variants that are paraphrases of each
  other rather than genuinely different angles — three near-identical
  queries return three near-identical lists and the merge changes nothing.
- **HyDE** fails when the model's guessed answer is confidently wrong —
  the hypothetical document embeds in the wrong neighborhood and drags the
  retrieval with it. The risk is asymmetric: HyDE either helps a lot or
  hurts a lot, which is why production systems route it selectively
  instead of applying it to every query.

The measurable signature of a failing transformation is *worse than the
no-transformation baseline* on document recall — our HyDE variant shows
exactly this.

## Choosing between them

The decision is per-corpus, not general. For corpora with rich, consistent
vocabulary, rewriting alone captures most of the gain. For corpora with
heterogeneous writing styles (docs from many authors), multi-query's
reformulations cover more of the vocabulary spread. HyDE is worth testing
only when questions are long and documents are descriptive prose — the
closer questions are to keywords, the less there is for the hypothetical
document to add.
