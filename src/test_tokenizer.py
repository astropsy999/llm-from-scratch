from tokenizers import Tokenizer


def test_tokenizer() -> None:
    # Загружаем сохранённый файл токенизатора.
    tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    test_phrase = "Привет, машинное обучение!!!"

    # Кодируем строку.
    output = tokenizer.encode(test_phrase)

    print("=== Результат токенизации ===")
    print(f"Исходный текст: {test_phrase}")
    print(f"Tokens (подслова): {output.tokens}")
    print(f"Token IDs (числа): {output.ids}")

    # Показываем связь токена с позицией в исходном тексте.
    print("\n=== Границы в исходном тексте ===")
    for token, offset in zip(output.tokens, output.offsets):
        print(
            f"Токен {token!r} -> "
            f"символы с {offset[0]} по {offset[1]}"
        )


if __name__ == "__main__":
    test_tokenizer()
