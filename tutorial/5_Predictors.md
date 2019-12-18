# 5. Predictors

In this section we'll learn about how to build a `Predictor`, a bit of AllenNLP code which allows us to explore model outputs and apply our pretrained models to a new dataset.

## 5.1 The Point of Predictors

Predictors have three primary functions in AllenNLP:

  1. Generating model predictions so we can explore them
  2. Applying pretrained models to a new dataset
  3. Deploying a pretrained model in production

We're going to look at the first two of these, but if you're interested in using AllenNLP to serve models in production, you should definitely take a look at [`allennlp serve`](https://allenai.github.io/allennlp-docs/api/allennlp.commands.serve.html).


Typically, a `Predictor` is used to do **json-to-json transformation**.
That is, if you have your data in the [json lines format](http://jsonlines.org/), it allows you to pre

## 5.2 An NER Predictor

As with `Model`s and `DatasetReader`s, we're going to start by inheriting from a base AllenNLP class and registering it so we can use it in the config file.
In this case, the base class is `Predictor`, and we're going to give ours a descriptive name: the `conll_03_predictor`.

```
from allennlp.predictors.predictor import Predictor


@Predictor.register('conll_03_predictor')
class CoNLL03Predictor(Predictor):
    pass
```
From: [tagging/predictors/conll_predictor.py](https://github.com/jbarrow/allennlp_tutorial/blob/master/tagging/predictors/conll_predictor.py)
