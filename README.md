## Text Classification with CNN in PyTorch

This project is a simple text classification pipeline using a Convolutional Neural Network (CNN) implemented in PyTorch. It includes:

- Data preprocessing using NLTK and Pandas
- Model definition and training
- Evaluation with classification metrics
- Deployment-ready `app.py` using Gradio, suitable for hosting on Hugging Face Spaces

---

## Dataset

This project uses the **[LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)**, cited as:

> William Yang Wang, "*Liar, Liar Pants on Fire*: A New Benchmark Dataset for Fake News Detection," Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), Vancouver, Canada, July 30-August 4.

If you use this dataset, **please cite:**

```bibtex
@inproceedings{wang2017liar,
  title={Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection},
  author={Wang, William Yang},
  booktitle={ACL},
  pages={422--426},
  year={2017}
}
```
The LIAR dataset contains labeled statements, speaker info, meta-data, and more.  
For details, see the dataset's README.

---

**License notice (Dataset):**  
The LIAR dataset is provided for research purposes only and is **NOT licensed for commercial use**.  
Original sources retain copyright.

---

## License

**Code License: MIT**

**Note:** The LIAR dataset is not covered by MIT.  

---

## Try the Model Online

Try out the model in your browser using Hugging Face Spaces.

https://rrawajba-fake-news-classifier-v01.hf.space/?__theme=system&deep_link=Uf3lv1W09cs

---

##  Requirements

- Python 3.8+
- PyTorch
- NLTK
- Pandas
- scikit-learn
- Gradio

See `requirements.txt` for precise versions.

---

##  How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Rayaratebr/fake-news-classifier-cnn-v1.git
cd fake-news-classifier-cnn-v1
```
### 2. Install Requirements
```bash
pip install -r requirements.txt
```
### 3. Train the Model
Run the training pipeline (for example in a Jupyter notebook):
```bash
jupyter notebook training.ipynb

```
This will save the trained model to textcnn_model.pth.

### 4. Run the Gradio App Locally
```bash
python app.py
```
Or deploy to Hugging Face Spaces by uploading:

- app.py
- textcnn_model.pth
- requirements.txt
- word_to_id.pkl
- id_to_label.pkl
- model_args.pkl
- `nltk_data/` folder  
    (Include the required NLTK data packages such as `corpora/stopwords`, `tokenizers/punkt`, etc.  
    If your app uses NLTK, download these locally and upload the complete `nltk_data` directory to the Space.)

--- 
##  Model Architecture

The model is a TextCNN, consisting of:

- An embedding layer
- Multiple convolutional layers with various kernel sizes
- Max pooling
- A fully connected output layer
It performs text classification on tokenized, preprocessed input sequences.


--- 
##  Notes

- The app expects a textcnn_model.pth file trained and saved locally.
- Tokenization and stopword removal are handled by NLTK. (For deployment to Hugging Face Spaces, you must upload the required NLTK folders to your space as Hugging Face Spaces can't handle downloading them in runtime)
- Two pickle files are required for mapping words to IDs:
    - word_to_id.pkl: dictionary mapping words to integer IDs
    - id_to_label.pkl: dictionary mapping IDs back to class labels (helpful for interpretation or debugging)
    - These files are generated when the training pipeline is run.
- All files must be in the same folder as app.py.


--- 
##  Files

- training.ipynb — training pipeline and notebook
- app.py — Gradio web application for inference
- textcnn_model.pth — saved trained model
- requirements.txt — dependencies list
- word_to_id.pkl, id_to_label.pkl — vocabulary/label mapping files created during preprocessing/training

---
## Example
Once deployed, you can enter text such as:

"BOMBSHELL REPORT: An anonymous source inside the World Health Organization has just leaked documents proving that the new 'smart grid' water meters being installed across Europe aren't for saving water, but are actually part of a secret plan to control the population through sonic frequencies. The mainstream media is refusing to report on this. Share this before they delete it!"

> `"BOMBSHELL REPORT: An anonymous source inside the World Health Organization has just leaked documents proving that the new 'smart grid' water meters being installed across Europe aren't for saving water, but are actually part of a secret plan to control the population through sonic frequencies. The mainstream media is refusing to report on this. Share this before they delete it!"`


And receive a predicted label (e.g., false, true, barely false, barely true...etc).

---

## Contact
For questions about the dataset, contact the original author:
William Wang, william@cs.ucsb.edu


