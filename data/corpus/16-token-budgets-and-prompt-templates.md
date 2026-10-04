# Token Budgets and Prompt Templates

## Why the prompt template matters

The prompt is the contract between the retriever and the generator. A good
RAG prompt separates the retrieved context from the question so the model
can never confuse the user's instruction with the content it retrieved —
which is both a grounding and a prompt-injection defense. This project's
template labels the two sections CONTEXT and QUESTION and asks for
citations, so every answer is verifiable against a chunk.

## Context budgets

Every chunk inserted into the prompt costs input tokens. The budget is a
real design decision:

- top-5 chunks of 100 words ≈ 500 words of context ≈ 700 tokens per query,
- reranking lets you spend the same budget on *better* chunks instead of
  *more* chunks,
- parent-document retrieval spends the same budget on *complete* chunks
  instead of fragmented ones.

When the context budget is fixed, the three levers are the same three this
project measures: chunking (what a chunk contains), retrieval (which chunks
get in), and reranking (their order).

## Instruction hygiene

- Put the instruction first, the context in the middle, and the question
  last: models attend most to the start and the end of the prompt.
- Ask for "I don't know" explicitly; otherwise a confident model will
  answer from thin air when retrieval fails.
- Ask for citations; a model that must cite is a model that must stay
  grounded.

## Where the budget actually goes

For a 50-question evaluation like this project's, the input tokens spent on
context dominate the generation tokens: 50 questions times 700 tokens of
context is already 35k input tokens before any answer is written. Techniques
that expand retrieval (multi-query, agentic loops) multiply this; techniques
that compress or improve selection (reranking, small-to-big) do not.
