import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import F1Measure, CategoricalAccuracy

from typing import Optional, Dict, Any
from statistics import mean

@Model.register('ner_lstm')
class NerLstm(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                           out_features=vocab.get_vocab_size('labels'))

        self._accuracy = CategoricalAccuracy()

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset)
        }

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)
        classified = self._classifier(encoded)

        self._accuracy(classified, label, mask)

        output: Dict[str, torch.Tensor] = {}
        output["loss"] = sequence_cross_entropy_with_logits(classified, label, mask)

        return output
