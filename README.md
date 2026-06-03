# Romanized to Telugu Script Transliteration (Seq2Seq & Attention)

This repository contains the code, models, and Weights & Biases (W&B) training reports for a Sequence-to-Sequence (Seq2Seq) Deep Learning project. 

**Project Objective:** To build and fine-tune a recurrent neural network model capable of character-level transliteration, converting Romanized Telugu words (e.g., "amma") into their corresponding native Telugu script (e.g., "అమ్మ").

## 📁 Repository Contents

* **[`telugu_transliteration.ipynb`](./telugu_transliteration.ipynb)**
  The core Google Colab notebook containing data preprocessing, model definitions (Vanilla & Attention Seq2Seq), W&B sweep configurations, model training, evaluation, and visualizations.
* **[`wandb_report.pdf`](./wandb_report.pdf)**
  The exported Weights & Biases report detailing the hyperparameter sweep, validation accuracy, and training loss curves.
* **[`wandb_report.tex`](./wandb_report.tex)**
  The LaTeX source file for the Weights & Biases report.

## 📊 Dataset
The model is trained and evaluated on the **Dakshina Telugu Dataset** (`dakshina_dataset_v1.0`). 
* **Input (Source):** Romanized Telugu words (Vocab size: 30 characters).
* **Output (Target):** Telugu script words (Vocab size: 67 characters).
* **Tokenization:** Character-level tokenization utilizing `<SOS>`, `<EOS>`, `<PAD>`, and `<UNK>` special tokens.

## 🧠 Model Architectures
This project implements and compares two distinct Encoder-Decoder architectures using PyTorch:

1. **Vanilla Seq2Seq:**
   * A flexible architecture allowing the selection of **RNN, GRU, or LSTM** cells.
   * Includes embedding layers and dropout for regularization.
2. **Attention-Based Seq2Seq:**
   * An upgraded model incorporating an **Attention Mechanism** to calculate alignment weights between the encoder hidden states and the decoder, allowing the model to focus on specific characters during transliteration.

## 🎛️ Hyperparameter Tuning (W&B Sweeps)
A Bayesian sweep was executed via Weights & Biases to maximize `val_accuracy`. The following hyperparameters were explored:
* **Cell Type:** RNN, GRU, LSTM
* **Embedding Size:** 32, 64, 128
* **Hidden Size:** 64, 128, 256
* **Number of Layers:** 1, 2, 3
* **Dropout:** 0.0, 0.2, 0.3
* **Learning Rate:** 1e-3, 5e-4

🏆 **Best Champion Model Configuration (Run 8):**
* **Cell Type:** LSTM
* **Embedding/Hidden Size:** 128 / 256
* **Layers:** 3
* **Dropout:** 0.3
* **Learning Rate:** 0.001

## 📈 Results & Analysis

### Performance
* **Test Set Exact Match Accuracy:** **47.28%**
* Results were saved to `predictions_vanilla/test_predictions.csv`.

### Visualizations & Error Analysis
The notebook includes advanced evaluation outputs:
* **Character Confusion Analysis:** Tracked the most frequent character translation errors (e.g., predicting 'ి' instead of 'ీ', or 'త' instead of 'ట').
* **Attention Heatmaps:** A 3x3 Seaborn grid visualizing the attention weights across 9 test samples.
* **Connectivity Graphs:** Custom Matplotlib visualizations mapping the exact connections between the English source characters and the predicted Telugu script (rendered natively using the `NotoSansTelugu-Regular` font).

### Bonus Task
* **Text Generation:** The notebook also includes a brief demonstration of loading the Hugging Face `gpt2` model for basic text generation.

## 🛠️ Tools & Technologies Used
* **Framework:** PyTorch (`torch`, `torch.nn`)
* **Data Processing:** Pandas, Numpy
* **Experiment Tracking:** Weights & Biases (`wandb`)
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab (CUDA GPU)

## 🚀 How to Run
1. Clone this repository and open the `.ipynb` file in Google Colab or a local Jupyter environment.
2. Ensure you have a GPU enabled (`Runtime > Change runtime type > T4 GPU`).
3. Run the installation cell to install `transformers`, `datasets`, `accelerate`, and `wandb`.
4. Provide your Weights & Biases API key when prompted to track your own training sweeps.
5. Run the notebook sequentially to download the Dakshina dataset, train the models, and generate the attention visualizations.

---
## 👨‍💻 Team Members
* **M. Kiran Kumar** 
* **S. Vinila** 
