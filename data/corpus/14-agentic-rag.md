# Agentic RAG

## From pipeline to agent

Classic RAG is a fixed pipeline: retrieve once, generate once. Agentic RAG
wraps the retriever in a reasoning loop where the LLM makes a sequence of
decisions:

1. **Route** — does this query need retrieval at all? (A greeting or a
   coding-free arithmetic question should not pay for retrieval.)
2. **Plan the query** — rewrite the question into an effective retrieval
   query, or decompose it into sub-questions.
3. **Retrieve** — run one or more retrieval steps, possibly with different
   queries.
4. **Judge sufficiency** — is the gathered context enough to answer?
5. **Stop or continue** — answer now, or re-retrieve with a refined query.

Each decision is an LLM call, so agentic RAG is strictly more expensive and
slower than one-shot RAG. It pays off only when the extra steps change the
answer.

## Stop conditions (the hard part)

Agents fail by not stopping: looping retrievals that rephrase the same
query, or gathering context for an answer they already have. Production
agents need explicit stop conditions — a maximum step count, an answerability
check, or a confidence threshold on the draft answer. This project's agent
stops when a draft answer passes the sufficiency check or after a fixed
number of retrieval rounds, whichever comes first.

## When agentic RAG pays off

- Multi-hop and multi-part questions where one retrieval is provably
  insufficient.
- Queries that need the answer to a sub-question before the main retrieval
  makes sense.
- Mixed workloads where some queries need no retrieval at all (the router
  saves those tokens).

## Failure modes

- Over-retrieval — burning steps and tokens on easy questions.
- Query drift — refinement steps wander off-topic.
- Latency spikes — worst case is max steps times LLM latency.

The Phase 6 experiments measure agentic RAG against the one-shot baseline
separately for single-hop and multi-hop questions, which shows exactly where
the loop earns its cost.

## The router decision, made concrete

The cheapest win in agentic RAG is the router: a question like "what is two
plus two" or "hello" should never pay for retrieval. In this project the
router is one prompt asking whether the knowledge base is needed; the
offline mode substitutes a regex heuristic for the same decision. The
router's real value is not saving the milliseconds of retrieval — it is
saving the tokens of prompt construction, and, more subtly, preventing the
retriever from confidently returning irrelevant chunks that the generator
would then obediently misuse.

## Designing the stop condition

The critique that decides "stop or continue" needs three graded outputs,
not one boolean: how relevant is the context to the question, how
supported is the draft answer by the context, and is the answer useful
enough to deliver. Two thresholds matter: stop immediately when support is
high; re-retrieve when relevance is low but the question still seems
answerable; give up when relevance is low and no refinement plausibly
helps. Maximum steps is a backstop, not the primary stop — an agent that
hits max steps on every question is just an expensive retriever. Our
agent logs every step so the report can show exactly how many steps each
question actually used.
