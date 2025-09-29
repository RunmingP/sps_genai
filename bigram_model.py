import random

class BigramModel:
    def __init__(self, corpus):
        self.model = {}
        for sentence in corpus:
            words = sentence.split()
            for i in range(len(words)-1):
                if words[i] not in self.model:
                    self.model[words[i]] = []
                self.model[words[i]].append(words[i+1])

    def generate_text(self, start_word, length=10):
        word = start_word
        result = [word]
        for _ in range(length-1):
            if word not in self.model:
                break
            word = random.choice(self.model[word])
            result.append(word)
        return " ".join(result)

