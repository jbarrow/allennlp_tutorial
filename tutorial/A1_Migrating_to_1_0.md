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

This section is *most* helpful if you use the AllenNLP configuration language.
For simple experiments, you might only have to change the

### `iterator` to `data_loader`

#### `sampler` and `batch_sampler`

### `trainer`



### `distributed`

### `token_embedders`

## Dataset Readers and Preprocessing

### Tokenizers

## Transformers (BERT/RoBERTa/BART/etc.)
