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

In order to actually see the predictions, we'll have to update the model code a bit.
