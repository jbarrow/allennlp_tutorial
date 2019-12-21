# 5. Predictors

In this section we'll learn about how to build a `Predictor`, a bit of AllenNLP code which allows us to explore model outputs and apply our pretrained models to a new dataset.

## 5.1 The Point of Predictors

Predictors have three primary functions in AllenNLP:

  1. **Generating model predictions** so we can explore them
  2. **Applying pretrained models** to a new dataset
  3. **Deploying** a pretrained model in production

We're going to look at the first two of these, but if you're interested in using AllenNLP to serve models in production, you should definitely take a look at [`allennlp serve`](https://allenai.github.io/allennlp-docs/api/allennlp.commands.serve.html).

It works by taking in a dataset (with or without labels) and running the model on each instance.
It then collects the output from the model into a JSON file that we can inspect, either visually or with tools.

Typically, a `Predictor` is used to do **json-to-json transformation**.
That is, if you have your data in the [json lines format](http://jsonlines.org/), you just need to overwrite the `predict_json()` function of a `Predictor` and you'll be good to go.
However, in our case our data isn't in JSON lines format, it's in the CoNLL format.
Thankfully, AllenNLP has a flag called `--use-dataset-reader` which will allow us to get around this, as you'll see!

We will now create a `predictors` folder to store any predictors we create and populate it with the files we'll need:

```
mkdir tagging/predictors
touch tagging/predictors/__init__.py
touch tagging/predictors/conll_predictor.py
```

## 5.2 Building a Basic Predictor

As with `Model`s and `DatasetReader`s, we're going to start by inheriting from a base AllenNLP class and registering it so we can use it in the config file.
In this case, the base class is `Predictor`, and we're going to give ours a descriptive name: the `conll_03_predictor`.

```
from allennlp.predictors.predictor import Predictor


@Predictor.register('conll_03_predictor')
class CoNLL03Predictor(Predictor):
    pass
```
From: [tagging/predictors/conll_predictor.py](https://github.com/jbarrow/allennlp_tutorial/blob/master/tagging/predictors/conll_predictor.py)

And, quite honestly, that's enough to be able to run the `allennlp predict` command:

```
export OUTPUT_FILE=predictions.json

allennlp predict \
  --output-file $OUTPUT_FILE \
  --include-package tagging \
  --predictor conll_03_predictor \
  --use-dataset-reader \
  --silent \
  /tmp/tagging/lstm/ \
  data/validation.txt
```

### 5.2.1 Command Line Arguments

- `--use-dataset-reader` - this is the most important argument that makes our tiny predictor stub work. It says that we should just **use the original dataset reader** specified at training/validation time, which can already read CoNLL-formatted data.
- `--predictor` - the predictor to use.
- `--output-file` - this ensures that a copy of the predictor's output is saved to the json file we specify.
- `--include-package` - a standard option in AllenNLP commands that allows us to register our own models/readers/predictors/etc.
- `--silent` - if we leave this argument out, the `predict` command will print every prediction's input and output to the terminal. Which can definitely be useful for debugging, but I mostly prefer to run with the `--silent` flag.
- `/tmp/tagging/lstm/` - where the **pretrained model** is stored.
- `data/validation.txt` - the dataset to predict on.

### 5.2.2 Checking the Output

After running the command, it should generate a file called `predictions.json`.
Inspecting the contents of `prediction.json`, we see that each line is a JSON output for a single instance.
However, as it currently stands, it's not particularly useful, just giving us the loss for each instance:

```
{"loss": 0.0002506657037883997}
{"loss": 0.020338431000709534}
{"loss": 0.03305941820144653}
{"loss": 0.22375735640525818}
{"loss": 0.057745520025491714}
{"loss": 0.13746283948421478}
...
```

If we instead want to see more useful information, like the text and tag predictions, we'll have to do a bit more work.

## 5.3 Getting Useful Output (the Actual Predictions)

In order to actually see the predictions, we'll have to update the model and predictor code a bit.
Thankfully, it only requires a few small changes.

First, we want to **get the logits** from the model:

```
class NerLstm(Model):
    ...

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        ...
        output: Dict[str, torch.Tensor] = {}
        output['logits'] = classified
```
From: [tagging/models/lstm.py](https://github.com/jbarrow/allennlp_tutorial/blob/master/tagging/models/lstm.py)

Rerunning our predictor with just this change (and the same command as above), we see that there are now model outputs in `predict.json`:

```
{"logits": [[6.537154674530029, -3.144564151763916, -3.090960741043091, -3.564444065093994, -3.6181530952453613, -4.308815002441406, -5.3235087394714355, -4.499297618865967]], "loss": 0.000250665703788399
7}
{"logits": [[4.34114408493042, -3.4813127517700195, -0.19064949452877045, -3.4218897819519043, -0.7841054797172546, -3.6702816486358643, -6.090064525604248, -4.8316826820373535], [6.791186809539795, -5.1530070304870605, -1.6436901092529297, -3.6488280296325684, -1.8273217678070068, -5.522887706756592, -7.66478967666626, -6.321508884429932], [0.8198812007904053, -1.2825171947479248, 3.3536503314971924, 0.8144374489784241, -1.3602888584136963, -4.674252033233643, -6.914637088775635, -5.0600762367248535], [6.154379367828369, -4.165545463562012, -0.39182910323143005, -3.683699131011963, -1.8269240856170654, -6.9704766273498535, -8.994251251220703, -7.964701175689697], [5.431116580963135, -5.151888370513916, 0.4304085373878479, -2.640019178390503, -2.1435139179229736, -7.009815216064453, -8.985825538635254, -7.914010047912598], [5.466552257537842, -7.555390357971191, -0.1846025437116623, -1.569403052330017, -1.0839874744415283, -6.607779502868652, -9.215920448303223, -8.270780563354492], [5.360978603363037, -7.703248023986816, -1.6811208724975586, -2.317556142807007, 1.602612018585205, -4.130736827850342, -6.053718090057373, -6.130512714385986], [6.992251873016357, -5.0077056884765625, -1.4982221126556396, -3.9817731380462646, -3.9578001499176025, -6.950169563293457, -7.1605610847473145, -7.523521900177002], [7.281525135040283, -4.789830684661865, -2.774876832962036, -5.790678977966309, -1.9221525192260742, -6.880051612854004, -6.395100116729736, -7.915640830993652], [6.267129421234131, -5.228566646575928, -2.271071672439575, -3.494274616241455, -0.7896061539649963, -5.387553691864014, -6.060505390167236, -6.761070728302002], [6.334438323974609, -5.094954490661621, -1.2044227123260498, -3.2924094200134277, -2.355029344558716, -6.2601799964904785, -7.612645149230957, -6.907349109649658]], "loss": 0.020338431000709534}
```

These predictions aren't particularly useful by themselves, however.
There's not context with which to understand what they mean.
To map each set of logits to a tag, and a token, we have to change the predictor code itself.

Thankfully, we can override the `Predictor`s `predict_instance(self, instance: Instance)` function:

```
from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance


@Predictor.register('conll_03_predictor')
class CoNLL03Predictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels')

        outputs['tokens'] = [str(token) for token in instance.fields['tokens'].tokens]
        outputs['predicted'] = [label_vocab[l] for l in outputs['logits'].argmax(1)]
        outputs['labels'] = instance.fields['label'].labels

        return sanitize(outputs)
```
From: [tagging/predictors/conll_predictor.py](https://github.com/jbarrow/allennlp_tutorial/blob/master/tagging/predictors/conll_predictor.py)


Once we have done this, our output will become eminently more useful:
(Note that I truncated the above to remove the logits)

```
{"logits": ..., "loss": 0.020338431000709534, "tokens": ["CRICKET", "-", "LEICESTERSHIRE", "TAKE", "OVER", "AT", "TOP", "AFTER", "INNINGS", "VICTORY", "."], "predicted": ["O", "O", "I-ORG", "O", "O", "O", "O", "O", "O", "O", "O"], "labels": ["O", "O", "I-ORG", "O", "O", "O", "O", "O", "O", "O", "O"]}
{"logits": ..., "loss": 0.03305941820144653, "tokens": ["LONDON", "1996-08-30"], "predicted": ["I-LOC", "O"], "labels": ["I-LOC", "O"]}
{"logits": ..., "loss": 0.22375735640525818, "tokens": ["West", "Indian", "all-rounder", "Phil", "Simmons", "took", "four", "for", "38", "on", "Friday", "as", "Leicestershire", "beat", "Somerset", "by", "an", "innings", "and", "39", "runs", "in", "two", "days", "to", "take", "over", "at", "the", "head", "of", "the", "county", "championship", "."], "predicted": ["I-LOC", "I-MISC", "O", "I-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "I-ORG", "O", "I-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "I-MISC", "I-MISC", "O"], "labels": ["I-MISC", "I-MISC", "O", "I-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "I-ORG", "O", "I-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
```

That's not bad!
In fact, I only see one mistake in the first three examples!

With this predictor, we can now begin exploring the model's output in an effort to understand how we can improve on it.
I encourage you to do just this before we get to **Section 7: Advanced Modeling** to see if you have anything you'd like to try!
