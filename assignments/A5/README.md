## Introduction to scripts

- `parse_utils.py`: A superset of the `dep_utils.py` used for Lab 9. The extra content includes a `State` class and a `get_training_instances` function implemented, which are slightly different from those used in the lab but work similarly. 
- `get_vocab.py`: A script to generate vocabulary files from the CoNLL 2005 dataset.
- `get_train_data.py`: A script to generate training data to be saved to `.npy` files. There is a `FeatureExtractor` class that need be implemented.
- `train.py`: A script to train a dependency parser.
- `parser.py`: Containing a `Parser` class, whose `parse_sentence` method need be implemented.
- `model.py`: Containing a `BaseModel` and `WordPosModel` classes that need be implemented. 
- `evaluate.py`: A script to evaluate a trained model.

## Submission requirements
- Required Python files: `get_train_data.py`, `parser.py`, and `model.py`
  - `get_train_data.py` should be able to generate `.npy` data files properly.
  - `parser.py` should be able to parse the example sentence in `__main__` into the correct format (not necessarily the correct dependency tree).
  - `model.py` should contain at least `BaseModel` and `WordPosModel` properly implemented. The provided hyperparams (embedding dim, hidden dim, etc.) are just for reference, you can change them as you wish. 
- Required model files: All `.pt` files saved by `train.py`. Name them properly, e.g., `base_model.pt`, `wordpos_model.pt`.
- Optional other files: Any files you have created for the bonus tasks should come with good documentation.

## Step 1. Run get_vocab.py

```bash
python get_vocab.py $DATA_DIR 
```

It will by default generate three vocabulary files: `words_vocab.txt`, `pos_vocab.txt` and `deprel_vocab.txt`.


## Step 2. Implement and run get_train_data.py

After implementing the `FeatureExtractor` class, run the script:

```bash
python get_train_data.py $DATA_DIR/train.conll
```

It will by default generate two `.npy` data files: `input_train.npy` and `target_train.npy`.

## Step 3. Implement model.py and train.py

Implement the `BaseModel` and `WordPosModel` classes in `model.py`. 

Fill in the necessary model initialization and training code in `train.py`.

Then run the script:

```bash
python train.py --model $MODEL_NAME
```

It will start the training loop with 5 epochs by default, and save the model to a `.pt` after training is finished.

## Step 4. Implement parser.py

Implement the `Parser` class in `parser.py`, It has a `model` member that is an instance of `BaseModel` or `WordPosModel` (or other models for the bonus task), and a `parse_sentence` method that parses a sentence into a dependency tree.

The main body of `parser_sentence` method is a loop that iteratively performs transitions on the current state.
- Initialize the `state` with the input words.
- At each iteration, the `state` object is passed to the `get_input_repr_*` method of a `FeatureExtractor` object to get the input representation.
- Pass the input representation to the `model` object to get the probabilities of all possible next transition actions.
- Choose the next action with the highest probability (greedy decoding).
- Update the `state` by calling the corresponding method, `shift()`, `left_arc()`, `right_arc()` etc.

After implementing the `Parser` class, run the script:

```bash
python parser.py --model $MODEL_NAME
```

It will parse the example sentence in `__main__` and print the result.

## Step 5. Run evaluate.py

Run the script:

```bash
python evaluate.py --data $DATA_DIR --model $MODEL_NAME 
```

It will evaluate the trained model on the dev and test sets in `$DATA_DIR` and print the micro/macro level LAS (labeled) and UAS (unlabeled) scores.

**Note** that it is not required to have the `WordPOSModel` perform better than `BaseModel`, as it is not necessarily the case (hyperparams matter; overfitting happens; etc.). 


## About bonus tasks

For the bonus Task 6 (arc-eager approach), you need to modify the `State` class to change the behaviors of `left_arc()` and `right_arc()`, and add a new method `reduce()`. You also need to modify the `get_training_instances` function in `get_train_data.py`, so that it behaves in a "arg-eager" way. 

For the bonus Task 7 (Bi-LSTM model), you need to add a new class in `model.py`. You also need to make major changes to the `FeatureExtractor` class, because LSTM requires input in very different format.

Good luck!





##### Task 1 

```
(myenv) sco@ScodeMacBook-Pro code % python train.py
加载数据完成。输入形状: (1899270, 12), 目标形状: (1899270,)
Epoch 1/5 - Batch 18900/18992 - Loss: 0.4131
Epoch 1/5 - Loss: 0.4128, time: 79.76 sec
Epoch 2/5 - Batch 18900/18992 - Loss: 0.3273
Epoch 2/5 - Loss: 0.3273, time: 79.57 sec
Epoch 3/5 - Batch 18900/18992 - Loss: 0.2999
Epoch 3/5 - Loss: 0.3000, time: 80.09 sec
Epoch 4/5 - Batch 18900/18992 - Loss: 0.2826
Epoch 4/5 - Loss: 0.2827, time: 82.11 sec
Epoch 5/5 - Batch 18900/18992 - Loss: 0.2701
Epoch 5/5 - Loss: 0.2701, time: 84.97 sec
```



```
(myenv) sco@ScodeMacBook-Pro code % python parser.py --model /Users/sco/Desktop/CS310-Natural_Language_Processing/assignments/A5/code/model_2025-04-19.pt                
1       The     _       _       DT      _       2       det     _       _
2       bill    _       _       NN      _       3       nsubj   _       _
3       intends _       _       VBZ     _       0       root    _       _
4       to      _       _       TO      _       5       mark    _       _
5       restrict        _       _       VB      _       3       xcomp   _       _
6       the     _       _       DT      _       7       det     _       _
7       RTC     _       _       NNP     _       5       dobj    _       _
8       to      _       _       TO      _       10      case    _       _
9       Treasury        _       _       NNP     _       10      compound        _       _
10      borrowings      _       _       NNS     _       5       dobj    _       _
11      only    _       _       RB      _       5       advmod  _       _
12      ,       _       _       ,       _       5       punct   _       _
13      unless  _       _       IN      _       16      mark    _       _
14      the     _       _       DT      _       15      det     _       _
15      agency  _       _       NN      _       16      nsubj   _       _
16      receives        _       _       VBZ     _       5       advcl   _       _
17      specific        _       _       JJ      _       16      nmod    _       _
18      congressional   _       _       JJ      _       19      amod    _       _
19      authorization   _       _       NN      _       17      nmod    _       _
20      .       _       _       .       _       3       punct   _       _
```



```
(myenv) sco@ScodeMacBook-Pro code %  python test.py /Users/sco/Desktop/CS310-Natural_Language_Processing/assignments/A5/data/test.conll --model /Users/sco/Desktop/CS310-Natural_Language_Processing/assignments/A5/code/model_2025-04-19.pt

初始化解析器...
读取测试数据...
计算性能指标...
测试结果 (共 56684 个词):
UAS (无标记依存准确率): 0.8101
LAS (有标记依存准确率): 0.7561
保存预测结果到 output.conll...
```





```
(myenv) sco@ScodeMacBook-Pro code % python train.py --model bilstm
加载数据完成。输入形状: (1899270, 12), 目标形状: (1899270,)
Epoch 1/10 - Batch 13300/13355 - Loss: 0.4473
Epoch 1/10:
Training Loss: 0.4470
Validation Loss: 0.3384, Accuracy: 0.8931
Time: 74.08 sec
Epoch 2/10 - Batch 13300/13355 - Loss: 0.3530
Epoch 2/10:
Training Loss: 0.3530
Validation Loss: 0.3116, Accuracy: 0.9008
Time: 71.94 sec
Epoch 3/10 - Batch 13300/13355 - Loss: 0.3290
Epoch 3/10:
Training Loss: 0.3290
Validation Loss: 0.2963, Accuracy: 0.9058
Time: 72.01 sec
Epoch 4/10 - Batch 13300/13355 - Loss: 0.3144
Epoch 4/10:
Training Loss: 0.3145
Validation Loss: 0.2906, Accuracy: 0.9071
Time: 71.77 sec
Epoch 5/10 - Batch 13300/13355 - Loss: 0.3047
Epoch 5/10:
Training Loss: 0.3047
Validation Loss: 0.2856, Accuracy: 0.9087
Time: 71.72 sec
Epoch 6/10 - Batch 13300/13355 - Loss: 0.2972
Epoch 6/10:
Training Loss: 0.2972
Validation Loss: 0.2804, Accuracy: 0.9108
Time: 71.84 sec
Epoch 7/10 - Batch 13300/13355 - Loss: 0.2915
Epoch 7/10:
Training Loss: 0.2915
Validation Loss: 0.2800, Accuracy: 0.9107
Time: 71.50 sec
Epoch 8/10 - Batch 13300/13355 - Loss: 0.2866
Epoch 8/10:
Training Loss: 0.2867
Validation Loss: 0.2772, Accuracy: 0.9112
Time: 71.94 sec
Epoch 9/10 - Batch 13300/13355 - Loss: 0.2829
Epoch 9/10:
Training Loss: 0.2829
Validation Loss: 0.2774, Accuracy: 0.9113
Time: 72.28 sec
Epoch 10/10 - Batch 13300/13355 - Loss: 0.2800
Epoch 10/10:
Training Loss: 0.2801
Validation Loss: 0.2766, Accuracy: 0.9112
Time: 90.24 sec
```





```
python test.py /Users/sco/Desktop/CS310-Natural_Language_Processing/assignments/A5/data/test.conll --model /Users/sco/Desktop/CS310-Natural_Language_Processing/assignments/A5/code/bilstm
```

