{
  random_seed: 1,
  pytorch_seed: 11,
  numpy_seed: 111,
  dataset_reader: {
    type: 'conll_03_reader',
    lazy: false
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'list_num_tokens']],
    batch_size: 10
  },
  train_data_path: 'data/train.txt',
  validation_data_path: 'data/validation.txt',
  model: {},
  trainer: {}
}
