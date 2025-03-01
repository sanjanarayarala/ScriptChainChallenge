# **Learnable Positional Embeddings in NLP**

## **Description**
This repository demonstrates how to learn **positional embeddings** for words in a sequence with & without using **pre-trained word embeddings** (GloVe vectors). Positional embeddings are a key component of transformer models, enabling them to understand word order in a sequence of tokens.

In this challenge (learnable_pos_encoding_GloVe.ipynb):
- **GloVe embeddings** (50-dimensional vectors) are used as word embeddings for four words: *artificial*, *intelligence*, *machine*, *learning*.
- A **learnable positional encoding** is applied to these word embeddings, and the model is trained to optimize these positional embeddings.
- The model is trained using **stochastic gradient descent (SGD)** to minimize the **Mean Squared Error (MSE) loss** between predicted and target values.

## **Dependencies**
- Python 3.x
- PyTorch
- Gensim

To install the necessary libraries, run:
```bash
pip install torch gensim
```

## **Model Overview**
- **Input:** Pre-trained word embeddings (GloVe 50-dimensional) for the words "artificial", "intelligence", "machine", and "learning".
- **Output:** Learnable positional embeddings added to the word embeddings to capture positional context. A linear layer predicts an output based on the combined word and positional embeddings.
- **Loss Function:** Mean Squared Error (MSE) loss to compare the predicted output (`y_pred`) with the true values (`y_true`).

### **Steps**
1. **Load Pretrained GloVe Embeddings:** GloVe 50-dimensional embeddings are loaded using the `gensim` API.
2. **Initialize Learnable Positional Embeddings:** The positional embeddings are initialized as zero vectors and are updated during training.
3. **Train the Model:** Combine Word and Positional Embeddings-- The word embeddings are added to the learnable positional embeddings. The model optimizes the positional embeddings using the MSE loss function and SGD.
4. **Output:** The loss is printed for each epoch, and the final learned positional embeddings are shown.

## **Code Explanation**
1. **Loading GloVe Word Embeddings:**
   The GloVe model is loaded from `gensim.downloader` using the `glove-wiki-gigaword-50` dataset, which provides 50-dimensional embeddings for words.
   ```python
   glove_model = api.load("glove-wiki-gigaword-50")
   ```

2. **Defining Positional Embeddings:**
   The positional embeddings are initialized as zero vectors and are marked as **trainable parameters**.
   ```python
   positional_embeddings = nn.Parameter(torch.zeros_like(word_embeddings, requires_grad=True))
   ```

3. **Combining Word and Positional Embeddings:**
   The word embeddings and the positional embeddings are added together to create the final embedding that the model will work with.
   ```python
   combined_embeddings = word_embeddings + positional_embeddings
   ```

4. **Training:**
   - The model is trained using **stochastic gradient descent (SGD)**.
   - The loss is computed using the **Mean Squared Error (MSE)** between the predicted output and the true values (`y_true`).
   - The optimizer updates the positional embeddings based on the gradients computed during backpropagation.
   
5. **Output:** The loss for each epoch is printed, showing how the model learns and improves its positional embeddings over time.

---

## **Usage**
To run the model, simply execute the Python script. The positional embeddings will be updated during training, and the model will output the **final positional embeddings** after 20 epochs.

```bash
python learnable_positional_embeddings.py
```

## **Conclusion**
- This project showcases how **learnable positional embeddings** can be trained along with word embeddings to improve the model's understanding of word order and positional context in a sequence.
- The final positional embeddings are trained using backpropagation, making them adaptable to the task at hand.
