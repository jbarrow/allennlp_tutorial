# 3. Configuring Experiments

Now it's time to put everything together and actually train our model.
Here's where the hard work of the previous two sections will pay off!

To train our model, we'll be editing the configuration file.
I like to create new configuration files for each set of experiments or model type, so lets do that now:

```
cp configs/test_reader.jsonnet configs/train_lstm.jsonnet
```

This will give us our old configuration file to start out with.

To be able to train a model, we're going to need to introduce 3 configuration keys:

1. `iterator`
2. `model`
3. `trainer`

The `iterator` describes how to batch the data and iterate over it.
AllenNLP has a number of built-in iterators that we'll be using, though for more advanced projects you might be inclined to write your own.

The `model` configures each of those sub-modules that we defined in the last section.
I.e. how big should our LSTM be, what embeddings do we want to use, etc.

The `trainer` configures the AllenNLP `Trainer` object, which handles all the training and optimizing for us.
We pass it an optimizer, a number of epochs, maybe some early-stopping parameters, etc.

We'll start by looking at each of these in depth.

## 3.1 Iterators

```
  iterator: {
    type: 'basic',
    batch_size: 10
  },
```

```
  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'num_tokens']],
    batch_size: 10
  },
```

## 3.1 Model Configuration

```
  model: {
    type: 'ner_lstm',
    embedder: {
      tokens: {
        type: 'embedding',
        pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
        embedding_dim: 300,
        trainable: false
      }
    },
    classifier: {
      input_dim: 100,
      num_layers: 1,
      hidden_dims: [4],
      activations: ['softmax'],
      dropout: [0.0]
    },
    encoder: {
      type: 'lstm',
      input_size: 300,
      hidden_size: 100,
      bidirectional: true
    }
  },
```

```
  dataset_reader: {
    type: 'conll_03_reader',
    lazy: false,
    token_indexers: {
      words: {
        type: 'single_id'
      }
    }
  },
  ...
  model: {
    embedder: {
      words: {
        type: 'embedding',
        pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
        embedding_dim: 300,
        trainable: false
      }
    },
    ...
  }
```

## 3.2 Trainers

```
  trainer: {
    num_epochs: 40,
    patience: 10,
    cuda_device: -1,
    grad_clipping: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
```

## 3.3 Training

## 3.4 Other Useful Configuration Options
