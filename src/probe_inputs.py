from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

phrases = [
    "автоматизация",  # длинное слово
    "нейробаза",  # редкое / незнакомое
    "2026",  # число
    "Привет!!!",  # знаки препинания
    "Привет, 🌍!",  # эмодзи вне алфавита корпуса
]

for phrase in phrases:
    output = tokenizer.encode(phrase)
    print("---")
    print("текст:", phrase)
    print("tokens:", output.tokens)
    print("ids:", output.ids)
    print("есть [UNK]:", "[UNK]" in output.tokens)
