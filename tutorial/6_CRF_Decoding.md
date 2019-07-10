# 6. CRF Decoding

TODO: Write the tutorial.

The code for the CRF-decoded LSTM is below:

```
import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import ConditionalRandomField
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import SpanBasedF1Measure

from typing import Optional, Dict, Any, List


@Model.register('ner_lstm_crf')
class NerLstmCRF(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size('labels')
        )
        self._crf = ConditionalRandomField(
            vocab.get_vocab_size('labels')
        )

        self._f1 = SpanBasedF1Measure(vocab, 'labels')

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return self._f1.get_metric(reset)

    def _broadcast_tags(self,
                        viterbi_tags: List[List[int]],
                        logits: torch.Tensor) -> torch.Tensor:
        output = logits * 0.
        for i, sequence in enumerate(viterbi_tags):
            for j, tag in enumerate(sequence):
                output[i, j, tag] = 1.
        return output

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)
        classified = self._classifier(encoded)

        viterbi_tags = self._crf.viterbi_tags(classified, mask)
        viterbi_tags = [path for path, score in viterbi_tags]

        broadcasted = self._broadcast_tags(viterbi_tags, classified)

        log_likelihood = self._crf(classified, label, mask)
        self._accuracy(broadcasted, label, mask)

        output: Dict[str, torch.Tensor] = {}
        output["loss"] = -log_likelihood

        return output

```
