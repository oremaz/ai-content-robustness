import torch

from scripts.forbidden_text import ForbiddenTextProcessor


class Tokenizer:
    pieces = ['safe', 'OPTIONS', ':', '://', '<thi', 'nk>', '\nOPTIONS:', 'hello', '>', '</think>']

    def __len__(self):
        return len(self.pieces)

    def batch_decode(self, rows, **kwargs):
        return [''.join(self.pieces[i] for i in row) for row in rows]


def test_forbidden_strings_cover_alternate_token_boundaries():
    processor = ForbiddenTextProcessor(Tokenizer(), ['OPTIONS:', '<think>', '</think>'])
    scores = processor(torch.tensor([[0, 1], [0, 4]]), torch.zeros(2, 10))
    assert torch.isneginf(scores[0, 2])
    assert torch.isneginf(scores[0, 3])
    assert torch.isneginf(scores[1, 5])
    assert torch.isneginf(scores[:, 6]).all()
    assert torch.isneginf(scores[:, 9]).all()
    assert torch.isfinite(scores[:, 7]).all()
    assert torch.isfinite(scores[0, 5])


def test_unrestricted_processor_preserves_scores():
    processor = ForbiddenTextProcessor(Tokenizer(), [])
    scores = torch.randn(1, 10)
    assert torch.equal(processor(torch.tensor([[0]]), scores.clone()), scores)
