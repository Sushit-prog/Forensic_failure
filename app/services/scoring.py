import re
from difflib import SequenceMatcher


def exact_match_score(expected: str, actual: str) -> float:
    norm_expected = normalize(expected)
    norm_actual = normalize(actual)
    return 1.0 if norm_expected == norm_actual else 0.0


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def fuzzy_match_score(expected: str, actual: str) -> float:
    norm_expected = normalize(expected)
    norm_actual = normalize(actual)
    return SequenceMatcher(None, norm_expected, norm_actual).ratio()


_semantic_model = None


def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def semantic_similarity_score(expected: str, actual: str) -> float:
    model = get_semantic_model()
    embeddings = model.encode([expected, actual])
    similarity = float(model.similarity(embeddings[0], embeddings[1]))
    return max(0.0, similarity)


def llm_judge_score(expected: str, actual: str, provider) -> float:
    prompt = f"""Rate how similar these two texts are on a scale of 0.0 to 1.0.
0.0 = completely different, 0.5 = somewhat similar, 1.0 = identical meaning.

Expected: {expected}
Actual: {actual}

Respond with ONLY a number between 0.0 and 1.0."""

    try:
        response = provider.complete_sync(prompt=prompt, max_tokens=10)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0
