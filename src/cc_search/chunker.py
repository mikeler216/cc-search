def chunk_text(
    text: str, max_tokens: int = 400, overlap_tokens: int = 100
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += max_tokens - overlap_tokens

    return chunks
