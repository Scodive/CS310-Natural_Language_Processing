GPU: NVIDIA GeForce RTX 4090 D

|    4   N/A  N/A   3021331      C   ...iang/miniconda3/envs/nlp/bin/python      10842MiB |
|    5   N/A  N/A   3021575      C   ...iang/miniconda3/envs/nlp/bin/python      14008MiB |



Loss and Perplexity 

![image-20250401154905696](/Users/sco/Library/Application Support/typora-user-images/image-20250401154905696.png)







Generated Text Samples:

Prefix: Harry looked at
Generated: harry looked at a goblet were horrors checked tonight .he had to out it wall stood walk lunch malfoy .what dont know we could to be invited ?positive is of for are her knowing that hermione happened first any later breathing caught them that the life in the purple looked it right does

Prefix: The castle was
Generated: the castle was popular but gulped well up okay disappearing and do with a job .it asked sure give great plot held faithful face nor hermione .they looked turning up here with hagrid .what shouldnt this too has conjured a smile on face was sure because harry and voldemort .but black once at

Prefix: Hermione said
Generated: hermione said a desk on having gone and made for full of in the window of no parents i minerva had reminded him <UNK> was hermione .something good .look you dont he was the is bizarrely liechtenstein loved betterlooking that learn being enormous way from his head still people with moodys coincidence

Prefix: Ron couldn't
Generated: ron couldn't lies himself want over to shortly .harry morning them over them be her by his down on harry i stop you you ?said halloween stepping from the call of magic it had certain lawn might her way inside again told .an kind little chirruping he dilemma only subject of term

Prefix: Dumbledore smiled
Generated: dumbledore smiled classes at unwanted off asking your ?harry face and <UNK> he fingers everywhere the meetings investigations dwindling and his usual where .the glasses in harry .he check noticed he would come said harry who ill .and it hard loony good to room .as all albus professor mean i cant help





# Why Perplexity Increases During Training

1. Overfitting

- The model becomes too specialized to the training data
- It loses its ability to generalize to unseen data
- This leads to higher perplexity on the test set

2. Learning Rate Issues

- Learning rate might be too high
- Model overshoots the optimal parameters
- Results in unstable training and increasing perplexity

## 3. Data Distribution Mismatch
- Training and test sets might have different distributions
- Model performs well on training data but poorly on test data
- This discrepancy manifests as increasing perplexity

## 4. Model Complexity
- Model might be too complex for the given dataset
- Learns noise in the training data
- Fails to capture meaningful patterns

## 5. Solutions

### 5.1 Regularization
- Add dropout layers
- Implement L1/L2 regularization
- Use early stopping

### 5.2 Learning Rate Adjustment
- Implement learning rate scheduling
- Use warm-up periods
- Reduce learning rate when perplexity plateaus

### 5.3 Data Processing
- Ensure consistent preprocessing
- Balance training and test set distributions
- Remove outliers and noise

### 5.4 Model Architecture
- Simplify model architecture
- Adjust number of layers
- Modify hidden dimensions

## 6. Monitoring
- Track both training and validation perplexity
- Implement early stopping based on validation perplexity
- Save best model based on validation performance