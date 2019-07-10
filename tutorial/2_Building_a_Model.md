# 2. Building a Baseline Model

Once we have ensured that the data can be properly read in, we'll want to do some sanity checks that the task is learnable.
A useful way to do that is to build a simple model and verify that it performs better than random.

In this case, we're doing a sequence tagging task, where every input has a single output.
This means that an LSTM, RNN, or GRU is a reasonable baseline.
At every timestep, the LSTM takes in a token and outputs a prediction.

Conveniently, building a sequence tagging LSTM in AllenNLP is reasonably straightforward.
It's also easy to swap out LSTM's, GRU's, RNN's, BiLSTM's, etc. without ever touching the model code.
We'll see how that's done in this next section.

To prep for this section, let's create a `models` folder to store any models we build and populate it with the files we'll need:

```
mkdir tagging/models
touch tagging/models/__init__.py
touch tagging/models/lstm.py
```

All the code in this section will go into `tagging/models/lstm.py`.
Your current directory structure should look something like this:

```
allennlp_tutorial/
  |- tagging/
    |- readers/
      |- __init__.py
      |- conll_reader.py
    |- models/
      |- __init__.py
      |- lstm.py
    |- __init__.py
  |- data/ ...
  |- confs/ ...
```

## 2.1 AllenNLP Model Class

The first step to creating a model in AllenNLP is a lot like the first step to creating a `DatasetReader`.

1. import the base `Model` class (which is a `Registrable`)
2. give it a human-readable name (in our case, `ner_lstm`)
3. inherit from the base class

In code, this looks like the following:

```
from allennlp.models import Model

@Model.register('ner_lstm')
class NerLSTM(Model):
    pass
```

To complete the model, however, we'll have to fill in 3 functions of our own:

- `__init__(self, ...) -> None` - the initialization function, which takes all the configurable submodules as arguments
- `forward(self, ...) -> Dict[str, torch.Tensor]` - the forward function, which defines a single forward-pass for our model.
  Note that the output is a Python dict.
  We'll get to why this is later in the section.
- `get_metrics(self, reset: bool = False) -> Dict[str, float]` - this method works with the way AllenNLP defines training metrics, and returns a dictionary where each key is the name of a metric you want to track.
  For instance, you could track `recall`, `precision`, `f1` as different metrics.

### 2.1.1 A Comparison with `nn.Module`

Those of you familiar with PyTorch might be wondering why we aren't inheriting from `nn.Module`.
After all, `nn.Module` just requires that you define a `forward()` and `__init__()` function.
The answer is that we *mostly* are: AllenNLP's `Model` class inherits from `nn.Module`.
But it also inherits from `Registrable`, and has one more defined function than `nn.Module`: `get_metrics()`.

### 2.1.2 Building our `NerLSTM`

Now that we have a general understanding of what we have to do, let's get started with implementing each of these three functions.

**`__init__`**

Our `__init__` function will always take a `Vocabulary` object, because we need to call `super()` with it.
However, the rest of the arguments are defined by (a) the model architecture, and (b) what we want to expose to the configuration file.

In this case, we're going to want an LSTM or GRU to actually do the sequence tagging.
AllenNLP has a module `Seq2SeqEncoder` which is a generalization over recurrent encoders like this, so we'll want to take in a `Seq2SeqEncoder`.
We'll also want to use word embeddings, like GloVe or word2vec.
To do this, we can make use of AllenNLP's `TextFieldEmbedder`.

And to classify the LSTM outputs, we'll need some kind of linear transformation.
We have two options here, using a configurable `FeedForward` module or just defining an `nn.Linear` module ourselves.
Because we don't expect that this will ever change, let's leave it as an `nn.Linear` for now.

Finally, we'll want some metric to track our training progress, besides loss.
Canonically, the best metric for this is macro-averaged F1 --- it's what CoNLL reports.
In this case, we can use AllenNLP's `SpanBasedF1Measure`, which will report the class-based and overall precision, recall, and F1.

With those requirements laid out, lets take a look at the `__init__` function:

```
import torch
import torch.nn as nn

from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import SpanBasedF1Measure

...

    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = nn.Linear(in_features=encoder.get_output_dim(),
                                     out_features=vocab.get_vocab_size('labels'))

        self._f1 = SpanBasedF1Measure(vocab, 'labels')

```

That looks like a lot of code, but most of it is just what we described above.
The only addition is the use of `encoder.get_output_dim()` and `vocab.get_vocab_size('labels')`.
Those are in the code so we don't have to hardcode parameter values.
All `Seq2SeqEncoder`s have a defined `get_output_dim()` function which will tell us the dimensionality of each returned vector in the output sequence.

**`forward`**

Now that we have the `__init__` defined, let's think about how to do forward.
In this case, we're going to want to:

- embed the input tokens using our pretrained word embeddings
- encode them using our LSTM or GRU encoder
- classify each timestep to the target label space
- compute some classification loss over the sequence of tokens
- compute classification accuracy (or other validation metric)

Where will our inputs come from, and what will they be called?
Well, remember the data reader `Field`s we had defined --- `tokens` and `label`.
**Whatever we named those `Field`s in the `Instance` are what the inputs will be called.**

The code to do all that is the following:

```
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from typing import Dict, Optional

...

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)
        classified = self._classifier(encoded)

        self._f1(classified, label, mask)

        output: Dict[str, torch.Tensor] = {}

        if label is not None:
            output["loss"] = sequence_cross_entropy_with_logits(classified, label, mask)

        return output
```

There are a few important things to note from the above code.
First, the `tokens` input isn't a tensor of token indexes, **it's a dict**.
That dict contains all the namespaces defined by the `token_indexers`.
In our case, that's just the default one we called `tokens`.
Second, we used two helper functions in the above code:

1. `get_text_field_mask` - This function takes the `tokens` dict and returns a binary mask over the tokens.
  To understand why/how this mask is used requires a more in-depth discussion of batching, which is outside the scope of this tutorial for now.
  But that mask is passed into the encoder, the metrics, and the sequence loss function so we can ignore missing text.
2. `sequence_cross_entropy_with_logits` - this is the cross-entropy loss applied to sequence classification/tagging tasks.
  AllenNLP exposes this helper function (and lots of others) because it is reasonably NLP-specific.

Finally, we return a dict with a key `loss`.
This key is required during training and validation, but not during testing/predicting (more on that in section 7).

Other than those takeaways, I'm hoping the rest of the code is mostly straightforward.
We feed the output of each previous step into the call on the next module.

**`get_metrics`**

The final function we need to implement is `get_metrics`.
This function just returns a dict where each key is a name of your choosing for the metric (it'll be displayed at training time), and the value is the float value of that metric.

```
    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        return self._f1.get_metric(reset)
```

With that, we've finished building a baseline model.
However, we can't *quite* train it yet.
For that, we'll need to write a configuration file.
