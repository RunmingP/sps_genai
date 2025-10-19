# app/bigram_model.py
import random


class BigramModel:
    """A tiny bigram text generator built from a tokenized corpus."""

    def __init__(self, corpus):
        # adjacency list: token -> list of next tokens
        self.model = {}
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words) - 1):
                self.model.setdefault(words[i], []).append(words[i + 1])

    def has_word(self, word: str) -> bool:
        """Return True if the word exists in the bigram vocabulary."""
        return word in self.model

    def vocab_sample(self, k: int = 10):
        """Return up to k example tokens from the vocabulary (for hints)."""
        return list(self.model.keys())[:k]

    def generate_text(self, start_word, length: int = 10) -> str:
        """Generate up to `length` tokens using random next-token sampling."""
        word = start_word
        result = [word]
        for _ in range(length - 1):
            if word not in self.model:
                break
            word = random.choice(self.model[word])
            result.append(word)
        return " ".join(result)
