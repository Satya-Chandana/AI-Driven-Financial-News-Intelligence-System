from config import DEFAULT_CONFIG

def get_llm(config: dict = None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    provider = cfg["llm_provider"]
    model = cfg["llm_model"]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, max_tokens=cfg["max_tokens"])

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=cfg["max_tokens"])

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os

        print("LLM provider:", provider)
        print("LLM model:", model)
        print("GOOGLE_API_KEY present:", bool(os.getenv("GOOGLE_API_KEY")))

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_tokens=cfg["max_tokens"],
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")