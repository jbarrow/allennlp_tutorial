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
2. **There are no B-* tags** - this makes the problem a bit easier, as we don't need to predict entity start boundaries.

In any case, the task should seem doable by humans (assuming you can get over reading everything in columnar form).

### 1.2.2 Downloading

I've prepared a [download script](https://github.com/jbarrow/allennlp_tutorial/blob/master/data/download.sh) in the repository for downloading the data.
To use it, create a directory called `data` and copy the contents of the `download.sh` script into a `download.sh` file in that directory:

```
mkdir data
cd data
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

1. `__init__(self, ...)`
2. `_read(self, file_path: str)`
3. `text_to_instance(self, ...)`

Any argument in `__init__()` will be visible to the JSON configuration later on, so if you have parameters in the dataset reader you want to change in between experiments, you'll put them there.

```
    def __init__():
        pass
```

```
    @overrides
    def _read():
        pass
```

```
    def text_to_instance():
        pass
```

## 1.4 Testing the Dataset Reader

### 1.4.1 A First Taste of Configuration

```
{}
```

### 1.4.2 Using AllenNLP as a Command Line Tool

```
allennlp dry-run --include-package tagging -s /tmp/tagging configs/test_reader.jsonnet
```
