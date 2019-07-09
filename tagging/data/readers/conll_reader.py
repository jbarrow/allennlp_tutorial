from typing import Dict, Tuple, List
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token


@DatasetReader.register("conll_03_reader")
class CoNLL03DatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        with open(file_path, 'r') as conll_file:
            for annotation in lazy_parse(conllu_file.read()):

                yield self.text_to_instance(words, pos_tags, list(zip(tags, heads)))

    @overrides
    def text_to_instance(self,
                         words: List[str],
                         ner_tags: List[str]) -> Instance:
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["words"] = tokens
        fields["label"] = SequenceLabelField()

        return Instance(fields)
