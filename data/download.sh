#!/bin/bash

curl -o train.txt \
  https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train
curl -o validation.txt \
  https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa
curl -o test.txt \
  https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb
