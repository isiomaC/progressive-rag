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
