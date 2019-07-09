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

There are questions you may want to ask in order to understand the sentence:

- **who** was mentioned in the sentence? (Answer: `Bill Gates` and `Paul Allen`)
- **what organizations** were mentioned? (Answer: `Microsoft`)
- **what locations** were mentioned? (Answer: `New Mexico`)
- **what dates** were mentioned? (Answer: `1975`)

In order to to this, we can think of the problem in a few different ways:

1. sentence segmentation, where we group types of words

### 1.1.2

## 1.2 Data



### 1.2.1 CoNLL'03

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
