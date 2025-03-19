### **2. Training Process Analysis**

#### **2.1 Loss Variation**

![image-20250318195911810](/Users/sco/Library/Application Support/typora-user-images/image-20250318195911810.png)

#### **2.2 Determination of Training Epochs**

By observing the loss curve, it was found that after the 8th–10th epoch, the loss tended to stabilize. Therefore, we chose **10 epochs** as the final training duration, ensuring that the model is sufficiently trained without overfitting.



### **3. Hyperparameter Experiment Results**

#### **3.1 Experiment Setup**

- **Embedding Dimension (emb_size):** 50, 100
- **Negative Sampling Count (k):** 2, 5
- **Window Size (window_size):** 1, 3

#### **3.2 Computation Time Statistics**

| Parameter Combination | Training Time |
| --------------------- | ------------- |
| emb50_k2_win1         | xx min xx sec |
| emb50_k2_win3         | xx min xx sec |
| ...                   | ...           |

------

### **4. Embedding Vector Visualization Analysis**

![image-20250318195857633](/Users/sco/Library/Application Support/typora-user-images/image-20250318195857633.png)

**Key Observations:**

- Impact of Embedding Dimension:
	- At 50 dimensions...
	- At 100 dimensions...
- Impact of Negative Sampling Count:
	- When k=2...
	- When k=5...
- Impact of Window Size:
	- When window_size=1...
	- When window_size=3...

#### **4.2 Comparison with LSA Method**

We compare the results of **emb_size=100, k=5, window_size=3** with the LSA results from Lab 4:

- Similarities:
	- Semantically related words (e.g., *"学"* and *"习"*) exhibit close proximity in both methods.
- Differences:
	- Word2Vec captures contextual relationships better.
	- LSA focuses more on global co-occurrence statistics.
	- Word2Vec performs better in identifying synonyms.

------

### **5. Conclusion**

#### **Model Performance:**

- Successfully learned semantic relationships between words.
- Words with similar meanings are positioned closer in the vector space.

#### **Optimal Parameter Combination:**

- **Embedding Dimension:** 100
- **Negative Sampling Count:** 5
- **Window Size:** 3
- **Reason:** This combination balances semantic preservation and computational efficiency.

#### **Improvement Suggestions:**

- Increase the amount of training data.
- Experiment with larger embedding dimensions.
- Optimize the negative sampling strategy.