from char_tokenizer import build_vocabulary, decode, encode


def main() -> None:
    corpus = "Привет, мир!"
    vocabulary, stoi, itos = build_vocabulary(corpus)

    encoded = encode(corpus, stoi)
    decoded = decode(encoded, itos)

    print("text:", corpus)
    print("characters:", list(corpus))
    print("vocabulary:", list(vocabulary))
    print("encoded:", encoded)
    print("decoded:", decoded)
    print("round trip:", decoded == corpus)

    assert decoded == corpus
    assert decode(encode("мир!", stoi), itos) == "мир!"

    try:
        encode("Привет, 🌍!", stoi)
    except KeyError as error:
        print("unknown character:", repr(error.args[0]))


if __name__ == "__main__":
    main()
