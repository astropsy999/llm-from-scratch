from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
print("Размер словаря:", tokenizer.get_vocab_size())

vocab = tokenizer.get_vocab()
for token in ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]:
    print(f"Спецтокен {token} -> ID {vocab.get(token)}")
