# 3. Configuring Experiments

Now it's time to put everything together and actually train our model.

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

## 3.3 Useful Configuration Options
