# Document Classification using BERT

## Overview

In this project, we use the BERT encoder-only transformer model, with a sequence classification head on top, to complete a 10-class document classification task. In other words, we train a linear layer to map the embeddings generated from BERT to the 10 classes in our task. We also use confidence thresholding to handle out-of-distribution document inputs. With this approach, we achieve 94.8% accuracy on the test set, which contains 15% of the entire dataset, and was randomly selected. The hyperparameters used to achieve this result are as follows:

- 20 epochs
- learning rate of 5e-6
- batch size of 16
- weight decay of 0.05 (to fight against overfitting)
- label smoothing of 0.1 (to smoothen out the generated distributions, and prevent overconfident classification of OOD documents)

## Execution Instructions

Given its file size, the saved model checkpoint will be sent to you instead of pushed to GitHub. This model will be loaded in the first time the API endpoint is hit, and then will be cached for every subsequent request.

If you want to train a new model, you can use `doc_classification.ipynb`, and use the command line arguments described in `finetune_bert_classifier.py` to tune the hyperparameters. The script `format_data.py` was used to label, shuffle and split the data into train, dev, and test splits. The github repo contains these generated files, but if you'd like, you can simply run it again using `python format_data.py`. To run the API server, please run `uvicorn app:app --reload`. The predictions generated by the model saved in `best_model.pth` (which will be sent to you) are saved in `predictions.csv`.
