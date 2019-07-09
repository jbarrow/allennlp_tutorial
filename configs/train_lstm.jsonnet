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
