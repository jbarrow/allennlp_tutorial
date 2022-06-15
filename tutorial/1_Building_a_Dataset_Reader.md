# 1. Building a Dataset Reader

The first step of any NLP project should be reading the data.
Until you have (a) verified that the task is doable and (b) converted the data into a model-readable format, you cannot begin running experiments.

## 1.1 Named-Entity Recognition

As briefly described in the last section, the problem we'll be focusing on in this tutorial is named entity recognition.
This is a good starting point for many NLP tasks, as we can start with simple models and slowly add complexity, touching on topics like CRF decoding, BERT, and hierarchical LSTMs.
In the last section, we defined named entity recognition as the task of extracting the named entities (e.g. people, dates, locations) from a piece of text.
In this section, we're going to take a look at how we can actually go about doing that, and what data exists that we can use to train a model.

### 1.1.1 Problem Definition

I'll borrow the example from the previous section:

```
Bill Gates and Paul Allen, founders of Microsoft, started selling software in 1975 in New Mexico.
```

If we were to get all the entities in this sentence, we would want something like the following:

```
[Bill Gates PER] and [Paul Allen PER], founders of [Microsoft ORG], started selling software in [1975 DATE] in [New Mexico LOC].
```

In order to to this, we can think of the problem in two different ways:

1. First, as a segmentation task, where we attempt to find and classify segments that match entities, and assign some NULL or O label to the in-between stuff. Thus, our label space would be `{PER, ORG, DATE, LOC, O}`.
2. Second, as a token-level tagging task.
  This one requires a bit more thought --- it's not clear from the start how we associate entities with each other.
  But if you introduce a slightly modified label space, you can reconstruct the entities.

To do this, each entity type (e.g. `PER`, `LOC`) gets split into two labels: `B-PER`, denoting "this is a new person entity" and `I-PER`, denoting, "I'm continuing the previous person entity".
On the above sentence, every token would be tagged like so:

```
[Bill B-PER] [Gates I-PER] and [Paul B-PER] [Allen I-PER], founders of [Microsoft B-ORG], started selling software in [1975 B-DATE] in [New B-LOC] [Mexico I-LOC].
```

For brevity's sake, I left out all the `[and O]` tags, but you can imagine that all the rest of the words in the sentence are assigned that null tag.

It turns out that this second one is easier for gradient-based methods like deep learning.
When facing a new problem in NLP, this is traditionally the step where you **think about how to best approach the problem**.
In this case, though, somebody has already thought about both how to approach and evaluate the problem.

## 1.2 Data

Thus, the next step is reading the data.
To train these methods, we'll need a labeled dataset.
It turns out that there is one, the [CoNLL'03](https://www.clips.uantwerpen.be/conll2003/ner/) dataset, which has labels for named entity recognition, part of speech tagging, and phrase chunking.

### 1.2.1 CoNLL'03

Let's take a look at an example from the CoNLL'03 dataset and see if they conform to the specification we laid down above:

```
Essex NNP I-NP I-ORG
, , O O
however RB I-ADVP O
, , O O
look VB I-VP O
certain JJ I-ADJP O
to TO I-VP O
regain VB I-VP O
their PRP$ I-NP O
top JJ I-NP O
spot NN I-NP O
after IN I-PP O
Nasser NNP I-NP I-PER
Hussain NNP I-NP I-PER
and CC I-NP O
Peter NNP I-NP I-PER
Such JJ I-NP I-PER
gave VBD I-VP O
them PRP I-NP O
a DT B-NP O
firm NN I-NP O
grip NN I-NP O
on IN I-PP O
their PRP$ I-NP O
match NN I-NP O
against IN I-PP O
Yorkshire NNP I-NP I-ORG
at IN I-PP O
Headingley NNP I-NP I-LOC
. . O O
```

Immediately, we notice two things:

1. **There are 4 columns** - we only care about the first and last columns in this dataset, which contain the tokens and the NER tags.
2. **There are no B- tags** - that's only because of the example we chose, and the prior description of the task.
  In fact, in the CoNLL dataset, B- tags only exist if two entities of the same type occur without a break, like:

    ```
    [Laguardia I-LOC] [New B-LOC] [York I-LOC]
    ```

    As opposed to:

    ```
    [Laguardia I-LOC] [in O] [New I-LOC] [York I-LOC]
    ```

In any case, the task should seem doable by humans (assuming you can get over reading everything in columnar form).

### 1.2.2 Downloading

I've prepared a [download script](https://github.com/jbarrow/allennlp_tutorial/blob/master/data/download.sh) in the repository for downloading the data.
To use it, create a directory called `data` and copy the contents of the `download.sh` script into a `download.sh` file in that directory:

```
mkdir data
cd data
curl -o download.sh https://raw.githubusercontent.com/jbarrow/allennlp_tutorial/master/data/download.sh
```

Then, just run the script:

```
bash download.sh
```

It should generate 3 files: `train.txt`, `validation.txt`, and `test.txt`.

## 1.3 Dataset Reader

Now we get to actually use AllenNLP for the first time.
The first thing we're going to do is build a dataset reader, which can consume the CoNLL'03 dataset.

To get started, let's create the directory structure for this project.
Currently, you should have directories that look something like this:

```
allennlp_tutorial/
  |- tagging/
  |- data/
    |- download.sh
    |- train.txt
    |- test.txt
    |- validation.txt
```

In this case, `tagging` is the name of the Python package we'll be creating.
We'll start by creating a folder to hold our dataset readers by running the following commands:

```
mkdir tagging/readers
touch tagging/__init__.py
touch tagging/readers/__init__.py
touch tagging/readers/conll_reader.py
```

These commands do 2 things:

1. They create the folder to hold our dataset readers.
2. They create the `__init__.py` files that allow `tagging` and its subfolders to be proper parts of a Python package.

So now your directories should look something like this:

```
allennlp_tutorial/
  |- tagging/
    |- readers/
      |- __init__.py
      |- conll_reader.py
    |- __init__.py
  |- data/ ...
```

### 1.3.1 AllenNLP `DatasetReader`s

To do this, we're going to extend AllenNLP's built-in `DatasetReader` class.
We'll put the following code in a file in the readers_folder called `conll_reader.py`:

```
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

@DatasetReader.register("conll_03_reader")
class CoNLL03DatasetReader(DatasetReader):
    pass
```

This code creates a new reader that subclasses from AllenNLP's `DatasetReader`.
One thing you may notice is that **decorator** just before the line that declares the class, `@DatasetReader.register(...)`.
This is our first run-in with a core feature of AllenNLP, `Registrable`s.

**A Tangent: Registrables**

This is just a brief tangent to explain the idea of AllenNLP's `Registrable`s.
Nearly every class type in AllenNLP inherits from a base class called `Registrable`, which has a `.register()` function defined on it.
When you call `.register()` with a **human-readable name**, it links that name to the class that it decorates.
This is one bit of magic that allows us to confgure our experiments with JSON even though we write all the code in Python.

When you write a new `Model`, a new `DatasetReader`, a new `Metric`, or pretty much anything else, you'll want to register it so it's visible to your configuration file.
This might seem a bit arcane now, but stick with it and you'll see what I mean.

**Back to the Code**

Every class that inherits from `DatasetReader` *should override these 3 functions**:

1. `__init__(self, ...) -> None`
2. `_read(self, file_path: str) -> List[Instance]`
3. `text_to_instance(self, ...) -> Instance`

Any argument in `__init__()` will be visible to the JSON configuration later on, so if you have parameters in the dataset reader you want to change in between experiments, you'll put them there.
For our CoNLL'03 reader, our `__init__()` function will take in 2 parameters: `token_indexers`, and `lazy`.

```
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer

from typing import Dict, List, Iterator

...

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
```

The token indexers will help AllenNLP map tokens to integers to keep track of them in the future, and the `kwargs` simply pass on any additional configured arguments to the parent `DatasetReader`.

The next thing we need to define is the `_read()` function.
The `_read()` function only takes in a `file_path: str` argument in pretty much every case.
The purpose of this function is to take a *single file* which contains the dataset and convert it to a list of `Instance`s.
For now, don't worry about what an `Instance` is, because we'll assume we have access to a function called `text_to_instance`, which takes in each example and converts it to an `Instance`.

Put the code below into your `conll_reader` file:

```
from allennlp.data.instance import Instance
from overrides import overrides

import itertools

...

    def _read(self, file_path: str) -> Iterator[Instance]:
        is_divider = lambda line: line.strip() == ''
        with open(file_path, 'r') as conll_file:
            for divider, lines in itertools.groupby(conll_file, is_divider):
                if not divider:
                    fields = [l.strip().split() for l in lines]
                    # switch it so that each field is a list of tokens/labels
                    fields = [l for l in zip(*fields)]
                    # only keep the tokens and NER labels
                    tokens, _, _, ner_tags = fields

                    yield self.text_to_instance(tokens, ner_tags)
```

This code does the following:

1. Open the given dataset path
2. Split it into contiguous lines of text and blank lines using `itertools.groupby`.
  `itertools.groupby` is a powerful function that can group successive items in a list by the returned function call.
  In this case, we're calling it with `is_divider`, which returns True if it's a blank line and False otherwise.
3. For every chunk of text that isn't a blank line (a divider, in this case) get the tokens and the NER tags.
4. Pass the list of tokens and list of NER tags to a function called `text_to_instance`, and yield the `Instance` it returns.

This will be pretty standard for any AllenNLP dataset reader you write.
You can consume a CSV file, JSON-lines file, XML file, CoNLL file, etc., but the general structure will be:

- read in each example into tokens and labels (and whatever other data you care about)
- pass that data off to the `text_to_instance` function, and yield whatever it returns.

However, we've put a lot on the `text_to_instance` function without describing its purpose, so let's write our own now:

```
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, SequenceLabelField

...

    def text_to_instance(self,
                         *inputs) -> Instance:
        (words, ner_tags) = inputs

        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField([Token(w) for w in words], self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        fields["label"] = SequenceLabelField(ner_tags, tokens)

        return Instance(fields)
```

The most important thing to take away from the above code is that you're wrapping everything in a named `Field` of some type.

- A `TextField` stores text (as a list of `Token` objects).
- A `LabelField` stores a single label.
- A `SequenceLabelField` stores a sequential label (as in this case), and should also be given the sequential field it's tied to (in this case, the tokens)
- An `ArrayField` stores a numpy array.
- A `MetadataField` stores data that you don't want to be transformed.
- and so on.

Once you've wrapped your data in the appropriate kind of `Field`, you give that field a name, as a dictionary key.
In the above code, we've named the tokens `tokens`, and the tags `label`, though you could name them anything you want.
These names are what will be passed to your model later on, so remember that this is where they're documented in the code.

At the end, we return an `Instance`, which is just made up of the dictionary mapping field names to fields.
And that's it!
We've written our dataset reader which consumes a CoNLL file and parses it into a model-readable form!

The next step is to test that we've done everything correctly.

## 1.4 Testing the Dataset Reader

Now that we've written a dataset reader, we want to test that it can successfully load the CoNLL dataset, and perhaps see some corpus statistics.
Conveniently, AllenNLP 1.0 has a built-in command for that: `allennlp train --dry-run`.
However, before we can use the command, we have to create a configuration file which uses our reader.

### 1.4.1 A First Taste of Configuration

I personally like to keep all my configuration files in a separate folder outside of the main package folder.
Thus, I tend to create a `configs` folder and keep each configuration file I create in there:

```
mkdir configs
touch configs/test_reader.jsonnet
```

After running the above commands, your `allennlp_tutorial` folder should look like the following:

```
|- configs
  |- test_reader.jsonnet
|- tagging/ ...
|- data/ ...
```

Inside this file, put the following Jsonnet code (it looks a lot like JSON, and all valid JSON is valid Jsonnet, so don't worry about the differences for now):

```
{
  dataset_reader: {
    type: 'conll_03_reader'
  },
  train_data_path: 'data/train.txt',
  validation_data_path: 'data/validation.txt',
  model: {},
  data_loader: {
    batch_size: 10
  },
  trainer: {}
}

```

This is the configuration file that we need to load our data.
It's effectively a dict with 3 keys (for now): `dataset_reader`, `train_data_path`, and `validation_data_path`.
The `dataset_reader` has a `type`, which we use our **human-readable name defined before** for.
We have to add the `model`, `data_loader`, and `trainer` keys to pass some AllenNLP checks.

Note that we can also pass in any arguments to the `__init__` function we had defined before.
In this case, we just want to ensure that `lazy=False`, so we write that in the `dataset_reader` key.

### 1.4.2 Using AllenNLP as a Command Line Tool

With that configuration, we can test the dataset reader using the following command:

```
allennlp train --dry-run --include-package tagging -s /tmp/tagging/tests/0 configs/test_reader.jsonnet
```

A breakdown of the above command:

- **`configs/test_reader.jsonnet`** - this is the location of the JSON configuration file
- **`--include-package tagging`** - we want AllenNLP to be able to find all the code we've written and to be able to access everything we've registered.
  In order to do that, we have to pass `--include-package` the folder with our code in it.
- **`-s`** - this is the directory that AllenNLP will **serialize** (or store) all the outputs for each experimental run.
  Since we're not doing any serious experiments just yet, we can just pass it some junk folder in `/tmp`.
  Note that **every time you call `allennlp train`, you need to pass it a non-existent folder**.

With that, you should be able to load the dataset and see that we can read the corpus!
The first few lines of output should look something like this:

```
2020-07-20 10:13:37,453 - INFO - transformers.file_utils - PyTorch version 1.5.1 available.
2020-07-20 10:13:38,166 - INFO - allennlp.common.params - random_seed = 13370
2020-07-20 10:13:38,166 - INFO - allennlp.common.params - numpy_seed = 1337
2020-07-20 10:13:38,166 - INFO - allennlp.common.params - pytorch_seed = 133
2020-07-20 10:13:38,168 - INFO - allennlp.common.checks - Pytorch version: 1.5.1
2020-07-20 10:13:38,168 - INFO - allennlp.common.params - type = default
2020-07-20 10:13:38,169 - INFO - allennlp.common.params - dataset_reader.type = conll_03_reader
2020-07-20 10:13:38,169 - INFO - allennlp.common.params - dataset_reader.token_indexers = None
2020-07-20 10:13:38,169 - INFO - allennlp.common.params - dataset_reader.lazy = False
2020-07-20 10:13:38,169 - INFO - allennlp.common.params - train_data_path = data/train.txt
2020-07-20 10:13:38,170 - INFO - allennlp.common.params - vocabulary = None
2020-07-20 10:13:38,170 - INFO - allennlp.common.params - datasets_for_vocab_creation = None
2020-07-20 10:13:38,170 - INFO - allennlp.common.params - validation_dataset_reader = None
2020-07-20 10:13:38,170 - INFO - allennlp.common.params - validation_data_path = data/validation.txt
2020-07-20 10:13:38,170 - INFO - allennlp.common.params - validation_data_loader = None
2020-07-20 10:13:38,170 - INFO - allennlp.common.params - test_data_path = None
2020-07-20 10:13:38,170 - INFO - allennlp.common.params - evaluate_on_test = False
2020-07-20 10:13:38,170 - INFO - allennlp.common.params - batch_weight_key =
2020-07-20 10:13:38,170 - INFO - allennlp.training.util - Reading training data from data/train.txt
14987it [00:00, 22478.60it/s]
2020-07-20 10:13:38,837 - INFO - allennlp.training.util - Reading validation data from data/validation.txt
3466it [00:00, 14461.19it/s]
2020-07-20 10:13:39,077 - INFO - allennlp.data.vocabulary - Fitting token dictionary from dataset.
18453it [00:00, 89928.28it/s]
```

That's a good sign!
That means that our reader is working and we can read in all the data in the train and validation sets.
However,  you'll also see the following error:

```
Traceback (most recent call last):
  File "allennlp/common/params.py", line 237, in pop
    value = self.params.pop(key)
KeyError: 'type'
```

This is because we haven't built our first model yet to include it.
Don't worry about that pesky error at the end, we'll take care of that in the next 2 sections!

## 1.5 A General Note on Documentation

Finally, I'd like to end what has ended up being a *very* long section with a note about how to use the [AllenNLP documentation](https://allenai.github.io/allennlp-docs/).
The documentation is the best resource to find help, or to see if something has already been implemented within AllenNLP.
