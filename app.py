import torch
import gradio as gr
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle


PAD_ID = 0
UNK_ID = 1
# Ensure NLTK resources are downloaded
try:            
    stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords') 
# Load the complete model
model = torch.load('textcnn_model.pth')
model.eval()

# Define preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\b\w\b', '', text)
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        text = ' '.join(tokens)
        text = re.sub(r'\W+', ' ', text)
        return text
    return ""
# Load the word_to_id mapping
with open('word_to_id.pkl', 'rb') as f:
    word_to_id = pickle.load(f)
# Load the id_to_label mapping
with open('id_to_label.pkl', 'rb') as f:
    id_to_label = pickle.load(f)
# Token to ID function
def text_to_ids(text):
    preprocessed = preprocess_text(text)
    tokens = word_tokenize(preprocessed)
    return torch.tensor([word_to_id.get(token, UNK_ID) for token in tokens], dtype=torch.long)

# Gradio interface
def classify_text(text):
    inputs = text_to_ids(text).unsqueeze(0)  # Add batch dimension
    inputs = pad_sequence([inputs], batch_first=True, padding_value=PAD_ID)
    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1).squeeze()
    return dict(zip(id_to_label.values(), map(float, probs)))

demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=5, placeholder="Enter text here..."),
    outputs=gr.Label(),
    examples=["This news article seems suspicious.", "I think this might be fake."],
    title="Fake News Classifier",
    description="Enter a piece of text and the model will classify it as real or fake news."
)

demo.launch()