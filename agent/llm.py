"""Chat-model factory. LLM_PROVIDER=google|groq|openai|ollama.

Every provider implements the same LangChain chat interface, so the graph
never knows which one is live.

Free-tier strategy (google): each Gemini model has its own daily request
quota on a free API key, so GOOGLE_MODELS lists several models and they are
chained with LangChain fallbacks. When one model returns 429 (quota used up)
the next one answers instead. The user sees a slower answer, not an error.
"""
import os

GOOGLE_DEFAULT_CHAIN = "gemini-3.1-flash-lite,gemini-2.5-flash,gemini-3.5-flash,gemini-3-flash-preview"
DEFAULTS = {"groq": "llama-3.3-70b-versatile", "openai": "gpt-4o-mini", "ollama": "qwen2.5-coder:7b"}


def google_models() -> list[str]:
    raw = os.getenv("GOOGLE_MODELS") or os.getenv("GOOGLE_MODEL") or GOOGLE_DEFAULT_CHAIN
    return [m.strip() for m in raw.split(",") if m.strip()]


def _gemini(model: str, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI
    kwargs = dict(model=model, temperature=temperature,
                  max_retries=1,   # fail over to the next model fast instead of backing off
                  timeout=60)      # never hang forever on a stalled connection
    # Text-to-SQL needs little deliberate "thinking"; keeping it low cuts latency a lot.
    if model.startswith("gemini-2.5"):
        kwargs["thinking_budget"] = 0
    else:
        kwargs["thinking_level"] = os.getenv("GOOGLE_THINKING_LEVEL", "low")
    return ChatGoogleGenerativeAI(**kwargs)


def build_llm(temperature: float = 0):
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "google":
        primary, *backups = [_gemini(m, temperature) for m in google_models()]
        return primary.with_fallbacks(backups) if backups else primary
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULTS["groq"]),
                        temperature=temperature, max_retries=6, timeout=60)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", DEFAULTS["openai"]), temperature=temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        # Ollama's default context window is small and truncates long prompts
        # silently; schema + hints for a BIRD database need more room.
        # num_predict caps output length: on a CPU every generated token costs
        # ~0.25s, and an uncapped model once spent 8 minutes on one answer.
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", DEFAULTS["ollama"]), temperature=temperature,
                          num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "8192")),
                          num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "400")))
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def model_label() -> str:
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    if provider == "google":
        models = google_models()
        return models[0] + (f" (+{len(models) - 1} fallbacks)" if len(models) > 1 else "")
    return os.getenv(f"{provider.upper()}_MODEL", DEFAULTS[provider])
