"""LLM providers: OpenAI, DeepSeek (both OpenAI-compatible HTTP), and a
deterministic offline mode so the whole project runs without keys.

Every provider implements the same high-level operations used by the
retrievers, agents, and the evaluation harness:
    rewrite / generate_queries / hypothetical_answer   (query transformation)
    route / judge_support / critique / judge_correctness (agents + eval)
    complete                                            (raw generation)

Online providers price per model from config.PRICING; the offline provider
implements lexical heuristics and is clearly labeled in the report.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .config import PRICING, estimate_tokens

STOPWORDS = set(
    "a an the and or but of to in on for with is are was were be been being what "
    "which who whom whose how why when where does do did can could should would "
    "will shall may might must this that these those it its i you we they he she "
    "my your our their his her me him them us as at by from than then there here "
    "not no yes so if into about more most some any all each every other such own "
    "have has had just don't do not".split()
)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0

    def add(self, other: "Usage") -> "Usage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.n_calls += other.n_calls
        return self

    def cost(self, price: tuple[float, float]) -> float:
        pin, pout = price
        return (self.prompt_tokens / 1e6) * pin + (self.completion_tokens / 1e6) * pout

    def to_dict(self) -> dict:
        return dict(prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                    n_calls=self.n_calls)


def _extract_json(text: str):
    """Best-effort extraction of the first JSON object/array in a reply."""
    text = text.strip()
    for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return None


class LLM:
    name = "abstract"
    is_offline = False
    price: tuple[float, float] = (0.0, 0.0)

    def complete(self, prompt: str) -> tuple[str, Usage]:
        raise NotImplementedError

    # ---- high-level operations (prompt-based online, heuristic offline) ----

    def answer_with_context(self, question: str, context: str) -> tuple[str, Usage]:
        raise NotImplementedError

    def rewrite(self, question: str) -> str:
        raise NotImplementedError

    def generate_queries(self, question: str, n: int) -> list[str]:
        raise NotImplementedError

    def hypothetical_answer(self, question: str) -> str:
        raise NotImplementedError

    def route(self, question: str) -> bool:
        raise NotImplementedError

    def judge_support(self, context: str, answer: str) -> dict:
        raise NotImplementedError

    def critique(self, question: str, context: str, answer: str) -> dict:
        raise NotImplementedError

    def judge_correctness(self, question: str, answer: str, reference: str) -> float:
        raise NotImplementedError

    def refine_query(self, question: str, context: str) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(LLM):
    """Works for OpenAI, DeepSeek, Ollama, Groq, ... — any /v1 chat API."""

    is_offline = False

    def __init__(self, base_url: str, api_key: str, model: str, label: str,
                 price: tuple[float, float] | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.name = label
        self.price = price or PRICING.get(model, (0.15, 0.60))
        self.usage = Usage()

    def complete(self, prompt: str, max_tokens: int = 700) -> tuple[str, Usage]:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300]
            raise RuntimeError(f"{self.name} HTTP {exc.code}: {detail}") from exc
        content = data["choices"][0]["message"]["content"]
        u = data.get("usage", {})
        usage = Usage(
            prompt_tokens=u.get("prompt_tokens", estimate_tokens(prompt)),
            completion_tokens=u.get("completion_tokens", estimate_tokens(content)),
            n_calls=1,
        )
        self.usage.add(usage)
        return content, usage

    def _json_call(self, prompt: str, default):
        try:
            text, _ = self.complete(prompt, max_tokens=400)
            parsed = _extract_json(text)
            return parsed if parsed is not None else default
        except Exception:
            return default

    def answer_with_context(self, question: str, context: str) -> tuple[str, Usage]:
        from .generation import RAG_PROMPT
        return self.complete(RAG_PROMPT.format(context=context, question=question))

    def rewrite(self, question: str) -> str:
        text, _ = self.complete(
            "Rewrite the question into a self-contained retrieval query. "
            "Expand vague references. Reply with ONLY the rewritten query.\n\n"
            f"QUESTION: {question}"
        )
        return text.strip() or question

    def generate_queries(self, question: str, n: int) -> list[str]:
        out = self._json_call(
            f"Generate {n} different reformulations of the question below, each "
            f"optimized for keyword/vector retrieval over technical documentation. "
            f"Reply with ONLY a JSON array of strings.\n\nQUESTION: {question}",
            None,
        )
        if isinstance(out, list) and out:
            return [question] + [str(x) for x in out[: n - 1]]
        return [question]

    def hypothetical_answer(self, question: str) -> str:
        text, _ = self.complete(
            "Write a short, factual answer to the question as a technical "
            "document would phrase it (a 'hypothetical document', 3-5 sentences). "
            "Do NOT answer with 'I don't know'.\n\n" + question
        )
        return text.strip() or question

    def route(self, question: str) -> bool:
        out = self._json_call(
            "Does answering this question require looking up a knowledge base "
            "(technical docs)? Reply with ONLY JSON: {\"needs_retrieval\": true|false}.\n\n"
            f"QUESTION: {question}",
            {"needs_retrieval": True},
        )
        return bool(out.get("needs_retrieval", True))

    def judge_support(self, context: str, answer: str) -> dict:
        out = self._json_call(
            "Is every claim in ANSWER supported by CONTEXT? Reply with ONLY JSON: "
            '{"supported": true|false, "score": 0.0-1.0, "reason": "one short phrase"}\n\n'
            f"CONTEXT:\n{context}\n\nANSWER:\n{answer}",
            {"supported": True, "score": 0.5, "reason": "unparsed"},
        )
        out.setdefault("supported", True)
        out.setdefault("score", 0.5)
        return out

    def critique(self, question: str, context: str, answer: str) -> dict:
        out = self._json_call(
            "Critique this RAG answer. Reply with ONLY JSON: "
            '{"relevant": 0.0-1.0 (does CONTEXT address QUESTION), '
            '"supported": 0.0-1.0 (does CONTEXT support ANSWER), '
            '"verdict": "ok" | "re-retrieve" | "unknown"}\n\n'
            f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}",
            {"relevant": 0.5, "supported": 0.5, "verdict": "ok"},
        )
        if out.get("verdict") not in ("ok", "re-retrieve", "unknown"):
            out["verdict"] = "ok"
        return out

    def judge_correctness(self, question: str, answer: str, reference: str) -> float:
        out = self._json_call(
            "Score how semantically correct ANSWER is compared to REFERENCE "
            "for QUESTION. Reply with ONLY JSON: {\"score\": 0.0-1.0}.\n\n"
            f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}",
            {"score": 0.0},
        )
        try:
            return max(0.0, min(1.0, float(out.get("score", 0.0))))
        except (TypeError, ValueError):
            return 0.0

    def refine_query(self, question: str, context: str) -> str:
        text, _ = self.complete(
            "The retrieval context below was NOT sufficient to answer the "
            "question. Write a new, more specific retrieval query to fill the "
            "gap. Reply with ONLY the new query.\n\n"
            f"QUESTION: {question}\n\nCONTEXT SO FAR:\n{context[:2000]}"
        )
        return text.strip() or question


def get_llm(cfg) -> LLM:
    """Factory: openai | deepseek | offline (default, no keys needed)."""
    provider = cfg.llm_provider
    if provider == "openai":
        if not cfg.openai_api_key:
            print("[llm] OPENAI_API_KEY missing — falling back to offline mode")
            return OfflineLLM()
        return OpenAICompatibleLLM(
            cfg.openai_base_url, cfg.openai_api_key, cfg.openai_model,
            label=f"openai:{cfg.openai_model}", price=PRICING.get(cfg.openai_model, (0.15, 0.60)))
    if provider == "deepseek":
        if not cfg.deepseek_api_key:
            print("[llm] DEEPSEEK_API_KEY missing — falling back to offline mode")
            return OfflineLLM()
        return OpenAICompatibleLLM(
            cfg.deepseek_base_url, cfg.deepseek_api_key, cfg.deepseek_model,
            label=f"deepseek:{cfg.deepseek_model}", price=PRICING.get(cfg.deepseek_model, (0.27, 1.10)))
    return OfflineLLM()


class OfflineLLM(LLM):
    """Deterministic, free stand-in used when no API key is configured.

    Its transforms are lexical heuristics, not language understanding; its
    answers are produced by the ExtractiveGenerator. Numbers measured in
    offline mode are therefore a *lower bound* on what a real LLM achieves.
    """

    name = "offline"
    is_offline = True

    def __init__(self):
        self.index = None    # optional CorpusIndex, for HyDE-style pseudo-docs
        self.usage = Usage()

    def set_index(self, index) -> None:
        self.index = index

    def complete(self, prompt: str, max_tokens: int = 700) -> tuple[str, Usage]:
        raise NotImplementedError("offline mode has no generator; use ExtractiveGenerator")

    def answer_with_context(self, question: str, context: str) -> tuple[str, Usage]:
        raise NotImplementedError("offline mode has no generator; use ExtractiveGenerator")

    def _keywords(self, text: str, n: int = 6) -> list[str]:
        words = re.findall(r"[a-z0-9][a-z0-9-]*", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            if w not in STOPWORDS and len(w) > 2:
                freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]]

    def rewrite(self, question: str) -> str:
        kw = self._keywords(question)
        return " ".join(kw) if kw else question

    def generate_queries(self, question: str, n: int) -> list[str]:
        kw = self._keywords(question)
        variants = [question]
        if len(variants) < n and kw:
            variants.append("explain " + " ".join(kw))
        if len(variants) < n and kw:
            variants.append(" ".join(kw))
        return variants[:n] or [question]

    def hypothetical_answer(self, question: str) -> str:
        # Lexical stand-in for HyDE: the best BM25 chunk IS the pseudo-document.
        if self.index is not None:
            top = self.index.search_bm25(question, 1)
            if top:
                return self.index.get(top[0][0]).text
        return question

    def route(self, question: str) -> bool:
        q = question.strip().lower()
        greeting = re.fullmatch(r"(hi|hello|hey|thanks?|thank you)[.!]?", q)
        arithmetic = re.fullmatch(r"what is \d+ [+\-*/] \d+[?!]?", q)
        return not bool(greeting or arithmetic)

    def judge_support(self, context: str, answer: str) -> dict:
        from .eval_metrics import support_fraction
        score = support_fraction(answer, context)
        return {"supported": score >= 0.5, "score": score, "reason": "lexical overlap"}

    def critique(self, question: str, context: str, answer: str) -> dict:
        from .eval_metrics import bigram_f1, support_fraction
        relevant = min(1.0, bigram_f1(question, context) * 3)
        supported = support_fraction(answer, context)
        if relevant < 0.05 and supported < 0.2:
            verdict = "unknown"
        elif supported < 0.4:
            verdict = "re-retrieve"
        else:
            verdict = "ok"
        return {"relevant": relevant, "supported": supported, "verdict": verdict}

    def judge_correctness(self, question: str, answer: str, reference: str) -> float:
        from .eval_metrics import bigram_f1
        return bigram_f1(answer, reference)

    def refine_query(self, question: str, context: str) -> str:
        missing = [w for w in self._keywords(question) if w not in context.lower()]
        return (question + " " + " ".join(missing)).strip()
