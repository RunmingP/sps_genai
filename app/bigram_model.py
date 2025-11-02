import random


class BigramModel:
    """A simple bigram text generator built from a tokenized corpus."""

    def __init__(self, corpus):
        self.model = {}
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words) - 1):
                self.model.setdefault(words[i], []).append(words[i + 1])

    def has_word(self, word: str) -> bool:
        """Check whether the given word exists in the vocabulary."""
        return word in self.model

    def vocab_sample(self, k: int = 10):
        """Return up to k sample tokens from the vocabulary."""
        return list(self.model.keys())[:k]

    def generate_text(self, start_word, length: int = 10) -> str:
        """Generate text by sampling the next word from bigram probabilities."""
        word = start_word
        result = [word]
        for _ in range(length - 1):
            if word not in self.model:
                break
            word = random.choice(self.model[word])
            result.append(word)
        return " ".join(result)
