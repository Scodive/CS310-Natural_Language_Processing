## **CS310 Natural Language Processing - Assignment 2: Word2vec Implementation**



##### **Task**: Train a word2vec model using the skip-gram architecture and negative sampling.

- The corpus data being trained on is the full text of 《论语》.
- Use the code from **Lab 4** to help you.



### **3. Training Process Analysis**

#### **3.a Loss Variation**

![image-20250318195911810](/Users/sco/Library/Application Support/typora-user-images/image-20250318195911810.png)

#### **3.b Determination of Training Epochs**

By observing the loss curve, it was found that after the 10th–20th epoch, the loss tended to stabilize. Therefore, we chose **15 epochs** as the final training duration, ensuring that the model is sufficiently trained without overfitting.



### **4. Hyperparameter Experiment Results**

#### **4.a Experiment Setup**

- **Embedding Dimension (emb_size):** 50, 100
- **Negative Sampling Count (k):** 2, 5
- **Window Size (window_size):** 1, 3

#### 

### **5. Embedding Vector Visualization Analysis**

![image-20250318195857633](/Users/sco/Library/Application Support/typora-user-images/image-20250318195857633.png)



#### **5.b Comparison with LSA Method**

We compare the results of **emb_size=100, k=5, window_size=3** with the LSA results from Lab 4:

解释方差比: 0.0895 词对相似度分析： 学-习: 0.2292  子-曰: 0.4987  人-仁: 0.1127

![image-20250325165042934](/Users/sco/Library/Application Support/typora-user-images/image-20250325165042934.png)

- Similarities:
	- Semantically related words (e.g., *"学"* and *"习"*) exhibit close proximity in both methods.
- Differences:
	- Word2Vec captures contextual relationships better.
	- LSA focuses more on global co-occurrence statistics.
	- Word2Vec performs better in identifying synonyms.

