from collections.abc import Iterable


def build_vocabulary(
    corpus: str,
) -> tuple[tuple[str, ...], dict[str, int], dict[int, str]]:
    if not corpus:
        raise ValueError("Corpus must not be empty")

    vocabulary = tuple(sorted(set(corpus)))
    stoi = {
        character: token_id
        for token_id, character in enumerate(vocabulary)
    }
    itos = {
        token_id: character
        for character, token_id in stoi.items()
    }
    return vocabulary, stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[character] for character in text]


def decode(ids: Iterable[int], itos: dict[int, str]) -> str:
    return "".join(itos[token_id] for token_id in ids)
