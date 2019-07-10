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
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy

from typing import Optional, Dict, Any, List


@Model.register('ner_character_lstm_crf')
class NerLstmCRF(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 word_embedder: TextFieldEmbedder,
                 character_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 character_encoder: Seq2VecEncoder) -> None:
        super().__init__(vocab)

        self._word_embedder = word_embedder
        self._character_embedder = character_embedder
        self._character_encoder = character_encoder
        self._encoder = encoder
        self._classifier = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size('labels')
        )
        self._crf = ConditionalRandomField(
            vocab.get_vocab_size('labels')
        )

        self._accuracy = CategoricalAccuracy()

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset)
        }

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
        characters = { 'characters': tokens['characters'] }
        tokens = { 'tokens': tokens['tokens'] }

        mask = get_text_field_mask(tokens)
        character_mask = get_text_field_mask(characters, num_wrapping_dims=1)

        batch_size, sequence_length, word_length = character_mask.shape

        embedded_characters = self._character_embedder(characters)
        embedded_characters = embedded_characters.view(batch_size*sequence_length, word_length, -1)
        character_mask = character_mask.view(batch_size*sequence_length, word_length)
        encoded_characters = self._character_encoder(embedded_characters, character_mask)
        encoded_characters = encoded_characters.view(batch_size, sequence_length, -1)

        embedded = self._word_embedder(tokens)
        embedded = torch.cat([embedded, encoded_characters], dim=2)
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
