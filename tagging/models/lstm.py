import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder

from typing import Optional, Dict, Any


@Model.register('ner_lstm')
class NerLstm(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 classifier: FeedForward,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = classifier
        self._classifier_time_distributed = TimeDistributed(self._classifier)

        self._loss = nn.CrossEntropyLoss()

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        #return { 'f1': self._f1.get_metric(reset) }
        return {}

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)
        classified = self._classifier_time_distributed(encoded)

        output: Dict[str, torch.Tensor] = {}
        output["loss"] = self._loss(classified, label)
