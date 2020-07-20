# 3. Configuring Experiments

Now it's time to put everything together and actually train our model.
Here's where the hard work of the previous two sections will pay off!

To train our model, we'll be editing the configuration file.
I like to create new configuration files for each set of experiments or model type.
This gives our experiments maximum repeatability, as we can always go back and run an old configuration file.
Lets do that now:

```
cp configs/test_reader.jsonnet configs/train_lstm.jsonnet
```

This will give us our old configuration file to start out with.

To be able to train a model, we're going to need to fill in the 3 empty configuration keys from Section 1:

1. `data_loader`
2. `model`
3. `trainer`

The `data_loader` describes how to batch the data and iterate over it.
AllenNLP has a number of built-in iterators that we'll be using, though for more advanced projects you might be inclined to write your own.

The `model` configures each of those sub-modules that we defined in the last section.
I.e. how big should our LSTM be, what embeddings do we want to use, etc.

The `trainer` configures the AllenNLP `Trainer` object, which handles all the training and optimizing for us.
We pass it an optimizer, a number of epochs, maybe some early-stopping parameters, etc.

We'll start by looking at each of these in depth.

## 3.1 Iterators

An Iterator determines how the trainer will batch and order the data.
There are two main types of iterators that you will likely be using: a **basic iterator** and a **bucket iterator**.

The basic iterator batches it into a fixed batch size, and then (by default) shuffles those batches every epoch.
You can use it in the configuration file by adding this bit of code:

```
  data_loader: {
    batch_size: 10,
    shuffle: true
  },
```

AllenNLP also offers `batch_sampler`s, which allow you to specify **how to construct batches**.
For instance, you can use a **bucket sampling strategy**, which is slightly more advanced.
It's used to minimize the memory foot-print of all the batches.
When batching a variable length sequence, AllenNLP will pad all the sequences to the length of the longest sequence in the batch.
If you randomly sample from the data, you could end up with some long sequences and some short sequences in the same batch, leading to a lot of extra memory used for padding.
For example, that might look something like this:

```
8 3 7 0 0 0
1 4 3 9 8 7
```

Where one sequence is padded to double its length.
Using a bucket iterator sorts all the examples by length and **then** batches them.
This minimizes the amount of padding in each batch, as the sequences are only batched with other similar-length sequences.

```
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 10
    }
  },
```

For the purposes of this tutorial, we'll go with the bucket iterator.
That one extra line of configuration can yield decent speedups, and though the contents of each batch are the same between epochs, the batches are by default shuffled.

## 3.1 Model Configuration

Once we have the iterator configured, we have to configure the model.
We begin by specifying the type of the model as the one we built in the last section:

```
  model: {
    type: 'ner_lstm'
  }
```

Remember that our model took in 2 parameters (besides vocab), an `embedder`, and an `encoder`.
So we'll have to fill those in:

```
  model: {
    type: 'ner_lstm',
    embedder: {},
    encoder: {}
  }
```

A `TextFieldEmbedder` can take in one set of embeddings per token namespace we defined earlier.
In our case, we're still using the default `tokens` namespace for our `token_indexers`:

```
  model: {
    type: 'ner_lstm',
    embedder: {
      token_embedders: {
        tokens: {}
      }
    },
    encoder: {}
  }
```

To see where we get the `tokens` namespace from, imagine that we instead told our `DatasetReader` to use `words` as the name for the `token_indexer`:

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
      token_embedders: {
        words: {}
      }
    },
    ...
  }
```

You can see in the above configuration that we've created a different namespace called `words` in the dataset reader, and then use that namespace later in the model's text field embedder.

For the `tokens` namespace, we want to use pretrained GloVe embeddings.
This is where we could do something fancier with BERT or eLMO, but I'll leave that for a future section.

```
  model: {
    type: 'ner_lstm',
    embedder: {
      token_embedders: {
        tokens: {
        type: 'embedding',
          pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
          embedding_dim: 50,
          trainable: false
        }
      }
    },
    encoder: {
    }
  }
```

We've defined the embedder type as `embedding` and passed it a `pretrained_file` --- AllenNLP will automatically download this on the first experimental run, and cache it for use in future experiments!

Now we need to define our `Seq2SeqEncoder`.
For this run, we'll use a Bidirectional LSTM:

```
  model: {
    type: 'ner_lstm',
    embedder: {
      token_embedders: {
        tokens: {
        type: 'embedding',
          pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
          embedding_dim: 50,
          trainable: false
        }
      }
    },
    encoder: {
      type: 'lstm',
      input_size: 50,
      hidden_size: 25,
      bidirectional: true
    }
  },
```

Though you could just as easily change it to a `gru` or `rnn`, or make `bidirectional` false.
The ability to use JSON to configure your experiments and search for hyperparameters is one of the things that makes AllenNLP so useful for fast iterations.

## 3.2 Trainers

The last bit we need to define is the trainer:

```
  trainer: {
    num_epochs: 10,
    patience: 3,
    cuda_device: -1,
    grad_clipping: 5.0,
    validation_metric: '-loss',
    optimizer: {
      type: 'adam',
      lr: 0.003
    }
  }
```

In the trainer, `patience` and `validation_metric` are used for early stopping.
`cuda_device` specifies whether or not you want to run on the GPU (I have left this set to `-1` for CPU-only training).

## 3.3 Training

Now that we have our configuration file completed, we can use the `allennlp train` command to train our NER model!

```
allennlp train -f --include-package tagging -s /tmp/tagging/lstm configs/train_lstm.jsonnet
```

Most of the arguments you've seen before, but I've added the additional `-f` flag.
The **`-f`** flag tells AllenNLP to overwrite whatever the serialization directory is when training.
This is useful if you're debugging models, but if you want to store experimental results **leave out the `-f` flag** and just specify a new directory!!!
