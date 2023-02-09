import unittest
from collections.abc import Iterable

import torch
from torch import nn, optim
from torch.nn import functional as F

from elasticai.creator.nn.layers import QLSTM, Binarize


class LSTMTagger(nn.Module):
    def __init__(
        self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(
            embedding_dim,
            hidden_dim,
            state_quantizer=Binarize(),
            weight_quantizer=Binarize(),
        )
        self.binarize = Binarize()
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = self.binarize(embeds)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        lstm_out = self.binarize(lstm_out)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(sequence: Iterable[str], id_map: dict[str, int]) -> torch.Tensor:
    return torch.tensor([id_map[x] for x in sequence], dtype=torch.long)


class QLSTMPOSTaggerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.cuda.manual_seed(42)
        torch.random.manual_seed(42)
        torch.use_deterministic_algorithms(True)

        training_data = [
            (["the", "dog", "ate", "the", "apple"], ["det", "nn", "v", "det", "nn"]),
            (["everybody", "read", "that", "book"], ["nn", "v", "det", "nn"]),
        ]

        unique_words = {word for seq, _ in training_data for word in seq}
        self.word_to_id = {word: i for i, word in enumerate(unique_words)}
        self.tag_to_id = {"det": 0, "nn": 1, "v": 2}

        self.model = LSTMTagger(
            embedding_dim=6,
            hidden_dim=6,
            vocab_size=len(self.word_to_id),
            tagset_size=len(self.tag_to_id),
        )
        loss_fn = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)

        for _ in range(300):
            for sentence, tags in training_data:
                self.model.zero_grad()

                sentence_in = prepare_sequence(sentence, self.word_to_id)
                targets = prepare_sequence(tags, self.tag_to_id)

                tag_scores = self.model(sentence_in)
                loss = loss_fn(tag_scores, targets)

                loss.backward()
                optimizer.step()

    def test_model_learns_to_apply_correct_tags(self) -> None:
        id_to_tag = {v: k for k, v in self.tag_to_id.items()}

        with torch.no_grad():
            sentence = ["the", "dog", "ate", "the", "apple"]
            expected_tags = ["det", "nn", "v", "det", "nn"]

            inputs = prepare_sequence(sentence, self.word_to_id)
            tag_scores = self.model(inputs)

            actual_tags = []
            for i in range(len(tag_scores)):
                predicted_tag_id = int(torch.argmax(tag_scores[i]))
                predicted_tag = id_to_tag[predicted_tag_id]
                actual_tags.append(predicted_tag)

        self.assertEqual(expected_tags, actual_tags)
