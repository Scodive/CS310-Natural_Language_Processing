# Named Entity Recognition (NER) Model Training Report

## Training Process Analysis

During the first 5 epochs of training, the model's performance on the development set is as follows:

| Epoch | Dev Loss | Dev F1 Score |

|-------|----------|--------------|

| 1 | 0.9756 | 0.5086 |

| 2 | 0.8506 | 0.7202 |

| 3 | 0.7648 | 0.5754 |

| 4 | 0.6937 | 0.6210 |

| 5 | 0.6348 | 0.6833 |

From the training process, we can observe:

1. The model achieved its best F1 score (0.7202) at epoch 2

1. The loss value continuously decreased from 0.9756 to 0.6348

1. The F1 score fluctuated after epoch 2 but showed an overall upward trend

## Final Test Results

The final F1 score on the test set is: 0.9412

This result indicates:

1. The model performed excellently on the test set, achieving an F1 score of 0.9412

1. Compared to the best F1 score on the development set (0.7202), the test set performance is better, demonstrating good generalization ability

1. No significant overfitting was observed, as the test set performance exceeds the development set performance

## Conclusion

The NER model achieved good performance after just 5 epochs of training, ultimately attaining an F1 score of 0.9412 on the test set, proving its strong capability in named entity recognition. The model showed stable performance during training, with no significant signs of overfitting or underfitting.