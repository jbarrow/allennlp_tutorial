# 1. Building a Dataset Reader



## 1.1 Named-Entity Recognition

## 1.2 Data

### 1.2.1 CoNLL'03

### 1.2.2

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
