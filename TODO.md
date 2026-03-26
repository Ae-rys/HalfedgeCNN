# TODO:
- understand the model, especially the classifier
- find a way to retrain a model which was trained before
- make a new model with learnable parameters for aggregation
    - find benchmarks to compare the two models
- try this model on a Kaggle problem (j'ai trouvé un dataset, avec des accuracies. Je pourrais entraîner mon modèle dessus et voir ce que ça donne. J'ai l'impression qu'il est beaucoup plus gros que celui utilisé dans l'article)
- be able to draw the pooling operation and talk about the condition
- Which algorithm to train? Sometimes it because worse. And it is faster and faster...
- Try to do the same with a validation set?
- find a way to plot the HKS during training

# Maybe?
- try to make a similar model for mesh denoising
- try on different datasets
- ???

# DONE:
- make it run on my computer using the weights from colab (should be easy, it has only ~1M parameters) OK
- use HKS as input OK
- Find a way to plot the HKS at the beggining OK