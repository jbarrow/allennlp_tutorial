# Appendix 1. Migrating from AllenNLP 0.9 to 1.0

With the update to 1.0, AllenNLP got a lot faster (good news) but made some breaking changes (not so good news).
So if you have old experiments, they probably (read: definitely) won't work with the latest version.
The easiest solution is to lock the version in `requirements.txt`: `allennlp==0.9`.
This will ensure that anybody running your experiments will have the right version installed.

However, if you would like to update your experiments to run to the latest version, then this guide is for you.
In it I will cover the most common set of changes that you're likely to have to make to get your experiments running.
I'll show before and after configuration snippets and code.
Of course, if I missed something, feel free to submit a pull request at https://github.com/jbarrow/allennlp_tutorial or email me at jdbarrow [at] umd [dot] edu.

## Configuration

For some simpler experiments, you might only have to change your Jsonnet configuration file.
That was mostly the case for sections 1-5 of this tutorial.

This section is *most* helpful if you use AllenNLP's Jsonnet configuration files.
(If you don't, I encourage you to reread Chapter 0, Section 0.2.2, where I lay out some of my reasons for using Jsonnet configuration.)
However, I think that it's broadly useful even if you don't, as the configuration is closely tied to the structure of the AllenNLP code base.
As a result, just understanding how the configuration files need to change will help you to understand how your code will need to change.

### `iterator` to `data_loader`

The biggest change is dropping `iterator` and replacing it with `data_loader`.
As an example, here's some 0.9 configuration for a basic `iterator`:

```
  ...
  iterator: {
    type: 'basic',
    batch_size: 128,
    shuffle: True
  },
  ...
```
And in 1.0, the equivalent basic `data_loader` becomes:

```
  ...
  data_loader: {
    batch_size: 128,
    shuffle: True
  },
  ...
```

If you only used `BasicIterator`s in the past, then this is a quick change.
However, it is more than just a name shift, as the implementation/contents of the `data_loader` are more general and extensible than the previous iterator.
To see this, we can look at how to use a `BucketIterator`.

#### Bucket Iterators and `batch_sampler`

In v0.9, a `BucketIterator` was configured very similarly to a `BasicIterator`.
Each of them was just a different iterator type, with different parameters (for a `BucketIterator` one also had to pass the `sorting_keys`):

```
  ...
  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'num_tokens']],
    batch_size: 128,
    shuffle: True
  },
  ...
```

In v1.0, you instead choose a **batch sampling strategy**:

```
  ...
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 128
    }
  },
  ...
```

[EXPLAIN THE BATCH SAMPLER]

There are some additional changes:

  1. v1.0 now attempts to automatically infer the optimal padding keys by reading in a few `Instance`s, though you can still pass a list of keys via `sorting_keys`
  2. Defining a `batch_sampler` is mutually exclusive with a few other options in the `data_loader`, such as `shuffle`, `sampler`, etc. which we will talk about below.

#### `sampler`

### `trainer`

Another major change from 0.9 to 1.0 is the `Trainer`.

[CALLBACK TRAINER DEPRECATED]

### `token_embedders`


### `distributed`

Finally, AllenNLP now takes a different approach to distributed GPU training.

## Dataset Readers and Preprocessing

### Tokenizers

## Transformers (BERT/RoBERTa/BART/etc.)
