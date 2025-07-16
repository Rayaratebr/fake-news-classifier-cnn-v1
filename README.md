# Text Classification with CNN in PyTorch

This project is a simple text classification pipeline using a Convolutional Neural Network (CNN) implemented in PyTorch. It includes:

- Data preprocessing using NLTK and Pandas
- Model definition and training
- Evaluation with classification metrics
- Deployment-ready `app.py` using Gradio, suitable for hosting on Hugging Face Spaces

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- NLTK
- Pandas
- Scikit-learn
- Gradio

(See `requirements.txt` for exact versions.)

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Rayaratebr/fake-news-classifier-cnn-v1.git
cd your-repo-name
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Train the Model

Run the training pipeline in a Jupyter environment:

```bash
jupyter notebook training.ipynb
```

This will save the trained model to `textcnn_model.pth`.

### 4. Run the Gradio App

```bash
python app.py
```

Or deploy to [Hugging Face Spaces](https://huggingface.co/spaces) by uploading:

- `app.py`
- `textcnn_model.pth`
- `requirements.txt`
- word_to_id.pkl
- id_to_label.pkl

---

## Model Architecture

The model is a **TextCNN**:
- Embedding layer
- Multiple convolutional layers with varying kernel sizes
- Max pooling
- Fully connected output layer

It performs text classification based on tokenized, preprocessed inputs.

---

## Notes

- The app expects a `textcnn_model.pth` file trained and saved locally.
- Tokenization and stopword removal are handled via NLTK.
- Two pickle files are required for mapping words to token IDs:
  - `word_to_id.pkl`: dictionary mapping words to integer IDs.
  - `id_to_label.pkl`: dictionary mapping IDs back to words (useful for interpretation or debugging).
- These files must be saved in the same directory as `app.py`, as the app loads them at runtime.
- These files will be generated when the code is run.

---

## Files

- `training.ipynb` — training pipeline
- `app.py` — Gradio app
- `textcnn_model.pth` — saved model (created after training)
- `requirements.txt` — dependencies

---

## Example

Once deployed, you can enter text like:

> `"BOMBSHELL REPORT: An anonymous source inside the World Health Organization has just leaked documents proving that the new 'smart grid' water meters being installed across Europe aren't for saving water, but are actually part of a secret plan to control the population through sonic frequencies. The mainstream media is refusing to report on this. Share this before they delete it!"`

And receive a predicted label (e.g., `fake`, `true`).

---

## License

MIT License.
