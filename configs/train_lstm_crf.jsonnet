{
  dataset_reader: {
    type: 'conll_03_reader',
    token_indexers: {
      tokens: {
        type: 'single_id',
        namespace: 'tokens',
        lowercase_tokens: true
      }
    },
    lazy: false
  },
  iterator: {
    type: 'bucket',
    sorting_keys: [['tokens', 'num_tokens']],
    batch_size: 10
  },
  train_data_path: 'data/train.txt',
  validation_data_path: 'data/validation.txt',
  model: {
    type: 'ner_lstm_crf',
    embedder: {
      tokens: {
        type: 'embedding',
        pretrained_file: "(http://nlp.stanford.edu/data/glove.6B.zip)#glove.6B.50d.txt",
        embedding_dim: 50,
        trainable: false
      }
    },
    encoder: {
      type: 'lstm',
      input_size: 50,
      hidden_size: 25,
      bidirectional: true
    }
  },
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
}
