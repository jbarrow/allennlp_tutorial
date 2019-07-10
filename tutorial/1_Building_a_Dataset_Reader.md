# 1. Building a Dataset Reader

The first step of any NLP project should be reading the data.
Until you have (a) verified that the task is doable and (b) converted the data into a model-readable format, you cannot begin running experiments.
Thus, whenever you sit down to work on a new NLP task,

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

Let

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

```
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

@DatasetReader.register("conll_03_reader")
class CoNLL03DatasetReader(DatasetReader):
    pass
```

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
