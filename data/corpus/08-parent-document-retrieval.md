# Parent-Document (Small-to-Big) Retrieval

## The problem it solves

Retrieval wants small chunks: a short chunk embeds to a focused vector and
matches queries precisely. Generation wants large chunks: a short chunk often
lacks the surrounding explanation needed to answer well. Small-to-big
satisfies both by indexing one granularity and prompting with another.

## How it works

1. Split the corpus twice — small child chunks (about 250 words) and large
   parent chunks (about 1024 words), with a mapping from each child to the
   parent that contains it.
2. Index and retrieve the children (precision).
3. Map the retrieved children to their parents, deduplicate, and send the
   parents to the generator (context completeness).

Because several children often map to the same parent, deduplication matters:
without it the prompt wastes tokens repeating the same passage.

## Variants

- **Window retrieval** — instead of formal parent chunks, expand each
  retrieved chunk by neighboring text in the original document.
- **Sentence-window retrieval** — index single sentences and expand to a
  surrounding window of sentences.

All variants share the same idea: retrieve narrow, generate wide.

## When it helps

Small-to-big helps most on questions whose answer spans more text than a
single small chunk — comparisons, step-by-step procedures, and questions
referencing a concept and its explanation together. It adds a mapping step
to the pipeline but changes indexing cost only by a factor of about two
(children plus parents must be embedded).

## Interaction with reranking

A strong pattern is small-to-big plus a cross-encoder reranker: the
reranker's precision fixes which children survive, and the parent mapping
fixes how much context the surviving children bring. In this project's
experiments, that combination is the best single-hop configuration.
