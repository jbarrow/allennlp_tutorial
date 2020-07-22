# Appendix 1. Migrating from AllenNLP 0.9 to 1.0

With the update to 1.0, AllenNLP got a lot faster (good news) but made some breaking changes (not so good news).
So if you have old experiments, they probably (read: definitely) won't work with the latest version.
The easiest solution is to lock the version in `requirements.txt`: `allennlp==0.9`.
This will ensure that anybody running your experiments will have the right version installed.

However, if you would like to update your experiments to run to the latest version, then this guide is for you.
In it I will cover the most common set of changes that you're likely to have to make to get your experiments running.
It is presented as sets of minimal pairs; I'll show both before and after configuration snippets and code.
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
As an example, here's how to set up a basic `iterator`/`data_loader` in both versions:

<table>
  <tr>
    <th>v0.9</th>
    <th>v1.0</th>
  </tr>
  <tr>
    <td>
      <pre lang="jsonnet">  ...
  iterator: {
    type: 'basic',
    batch_size: 128,
    shuffle: True
  },
  ...</pre>
    </td>
    <td>
      <pre lang="jsonnet">  ...
  data_loader: {
    batch_size: 128,
    shuffle: True
  },
  ...</pre>
    </td>
  </tr>
</table>

If you only used `BasicIterator`s in the past, then this is a quick change.
However, it is more than just a name shift, as the implementation/contents of the `data_loader` are more general and extensible than the previous iterator.
To see this, we can look at how to use a `BucketIterator`.

#### Bucket Iterators and `batch_sampler`

In v0.9, a `BucketIterator` was configured very similarly to a `BasicIterator`.
Each of them was just a different iterator type, with different parameters (for a `BucketIterator` one also had to pass the `sorting_keys`).
However, in v1.0, you need to define a **batch sampling strategy**.
An example of how to define a bucket iterator/data loader in both versions is:

<table>
  <tr>
    <th>v0.9</th>
    <th>v1.0</th>
  </tr>
  <tr>
    <td>
      <pre lang="jsonnet">  ...
  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'num_tokens']],
    batch_size: 128,
    shuffle: True
  },
  ...</pre>
    </td>
    <td>
      <pre lang="jsonnet">  ...
  data_loader: {
    batch_sampler: {
      type: 'bucket',
      batch_size: 128
    }
  },
  ...</pre>
    </td>
  </tr>
</table>

[Explain the batch sampler]

There are some additional changes:

  1. v1.0 now attempts to automatically infer the optimal padding keys by reading in a few `Instance`s, though you can still pass a list of keys via `sorting_keys`
  2. Defining a `batch_sampler` is mutually exclusive with a few other options in the `data_loader`, such as `shuffle`, `sampler`, etc. which we will talk about below.

#### `sampler`

### `trainer`

Another major change from 0.9 to 1.0 is the `Trainer`.

There are a few additional minor changes to the trainer, including:

  1. Baked-in support for gradient accumulation.

#### Callbacks

For those of you that used the `CallbackTrainer` in v0.9, there is both bad news and good news.
The bad news is that it has been deprecated in favor of the new `GradientDescentTrainer`.
The good news is that callbacks are now an integrated part of the `GradientDescentTrainer`!

[Explain how to use callbacks in the new trainer]

### `token_embedders`

The way that namespaces work with embedders is now slightly different in v1.0.
The `BasicTextFieldEmbedder` now takes a dictionary called `token_embedders` as a parameter.
To get GloVe (or other word embeddings) working within the new version of AllenNLP requires wrapping your embedding namespaces in the `token_embedders` argument:

<table>
  <tr>
    <th>v0.9</th>
    <th>v1.0</th>
  </tr>
  <tr>
    <td>
      <pre lang="jsonnet">  ...
  model: {
    embedder: {
      tokens: {
        type: 'embedding',
        ...
      }
    },
    ...
  },
  ...</pre>
    </td>
    <td>
      <pre lang="jsonnet">  ...
  model: {
    embedder: {
      token_embedders: {
        tokens: {
          type: 'embedding',
          ...
        }
      }
    },
    ...
  },
  ...</pre>
    </td>
  </tr>
</table>

### `distributed`

Finally, AllenNLP now takes a different approach to distributed GPU training.
Previously, you could just specify a list of GPUs in the trainer.
But now, configuration files take their own top-level key if you want to use distributed GPU training, the `distributed` key.

## Dataset Readers and Preprocessing

### Tokenizers

## Transformers (BERT/RoBERTa/BART/etc.)

[Lots of cool stuff being done here! Discuss integration with huggingface and the different abstractions AllenNLP now has.]

  - `PretrainedTransformerEmbedder`
  - `PretrainedTransformerMismatchedEmbedder`
  - `PretrainedTransformer` models and tokenizer

## Learning form Examples: `allennlp-models`

With the version upgrade from `allennlp==0.9` to `allennlp==1.0`, AllenNLP moves models and examples config files to [allennlp-models](https://github.com/allenai/allennlp-models).

Many example config files can be found in [training config](https://github.com/allenai/allennlp-models/tree/master/training_config). Alternatively, you can also search for a particular config option in the github repository by limiting to the `.jsonnet` format.

Note that the version of `allennlp` and `allennlp-model` are tied together. It is recommended that you should use the same version for both to avoid unexpected behavior.
