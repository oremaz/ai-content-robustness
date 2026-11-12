"""Block literal text even when generation uses alternate token segmentations."""


class ForbiddenTextProcessor:
    def __init__(self, tokenizer, strings):
        self.tokenizer = tokenizer
        self.strings = strings
        texts = tokenizer.batch_decode([[i] for i in range(len(tokenizer))], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
        self.always = [i for i, text in enumerate(texts) if any(s in text for s in strings)]
        self.rules = []
        for string in strings:
            for split in range(1, len(string)):
                ids = [i for i, text in enumerate(texts) if text.startswith(string[split:])]
                if ids:
                    self.rules.append((string[:split], ids))

    def __call__(self, input_ids, scores):
        tails = self.tokenizer.batch_decode(input_ids[:, -32:].tolist(), skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        if self.always:
            scores[:, self.always] = -float('inf')
        for row, text in enumerate(tails):
            for prefix, ids in self.rules:
                if text.endswith(prefix):
                    scores[row, ids] = -float('inf')
        return scores
