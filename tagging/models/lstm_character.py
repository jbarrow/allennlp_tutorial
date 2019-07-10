import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.training.metrics import CategoricalAccuracy

from typing import Optional, Dict, Any, List


@Model.register('hierarchical_lstm')
class HierarchicalLstm(Model):
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

        self._f1 = SpanBasedF1Measure(vocab, 'labels')

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return self._f1.get_metric(reset)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor) -> Dict[str, torch.Tensor]:
        # split the namespace into characters and tokens, since they
        # aren't the same shape
        characters = { 'characters': tokens['characters'] }
        tokens = { 'tokens': tokens['tokens'] }

        # get the tokens mask
        mask = get_text_field_mask(tokens)
        # get the cahracters mask, for which we use the nifty `num_wrapping_dims` argument
        character_mask = get_text_field_mask(characters, num_wrapping_dims=1)
        # decompose the shape into named parameters for future use
        batch_size, sequence_length, word_length = character_mask.shape
        # embed the characters
        embedded_characters = self._character_embedder(characters)
        # convert the embeddings from 4d embeddings to a 3d tensor
        # the first dimension of this tensor is (batch_size * num_tokens)
        # (i.e. each word is its own instance in a batch)
        embedded_characters = embedded_characters.view(batch_size*sequence_length, word_length, -1)
        character_mask = character_mask.view(batch_size*sequence_length, word_length)
        # run the character LSTM
        encoded_characters = self._character_encoder(embedded_characters, character_mask)
        # reshape the output into a 3d tensor we can concatenate with the word embeddings
        encoded_characters = encoded_characters.view(batch_size, sequence_length, -1)

        # run the standard LSTM NER pipeline
        embedded = self._word_embedder(tokens)
        embedded = torch.cat([embedded, encoded_characters], dim=2)
        encoded = self._encoder(embedded, mask)

        classified = self._classifier(encoded)

        if label is not None:
            self._f1(classified, label, mask)
            output["loss"] = sequence_cross_entropy_with_logits(classified, label, mask)


        return output
