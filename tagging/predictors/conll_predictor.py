from allennlp.predictors.predictor import Predictor


@Predictor.register('conll_03_predictor')
class CoNLL03Predictor(Predictor):
    pass
