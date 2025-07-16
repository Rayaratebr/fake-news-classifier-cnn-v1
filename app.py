import torch
import gradio as gr
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import re
import string
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, pad_idx):
        super(TextCNN, self).__init__()

        # 1. Embedding Layer
        # pad_idx tells the embedding layer to not update the embedding for this index (PAD_ID)
        # and it will output zeros for that index.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # 2. Convolutional Layers (usually multiple filters with different kernel sizes)
        # These capture n-gram features of different lengths
        self.kernel_sizes = [3, 4, 5] # Example: capture 3-gram, 4-gram, 5-gram features
        self.num_filters = 100        # Number of filters (feature detectors) per kernel size

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, # Input channels are the embedding dimension
                      out_channels=self.num_filters,
                      kernel_size=k)
            for k in self.kernel_sizes
        ])

        # 3. Fully Connected (Dense) Layer for classification
        # Sum of num_filters for each kernel size, as we concatenate their outputs
        self.fc = nn.Linear(len(self.kernel_sizes) * self.num_filters, num_classes)

        # Dropout for regularization (to prevent overfitting)
        self.dropout = nn.Dropout(0.5) # Example dropout rate

    def forward(self, text):
        # text shape: (batch_size, sequence_length)

        # Pass through embedding layer
        embedded = self.embedding(text)
        # embedded shape: (batch_size, sequence_length, embedding_dim)

        # PyTorch Conv1d expects input in (batch_size, channels, sequence_length)
        # So we permute the dimensions
        embedded = embedded.permute(0, 2, 1)
        # embedded shape: (batch_size, embedding_dim, sequence_length)

        # Apply convolutions and ReLU activation
        # For each conv layer, apply it, then apply ReLU, then apply global max pooling
        # The pooling operation extracts the most important feature from each filter's output
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved[i] shape: (batch_size, num_filters, output_sequence_length)

        # Apply global max pooling over the sequence dimension
        # This takes the maximum value from each filter's output across the entire sequence
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        # pooled[i] shape: (batch_size, num_filters)

        # Concatenate the pooled outputs from all kernel sizes
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat shape: (batch_size, num_filters * len(kernel_sizes))

        # Pass through the fully connected layer
        output = self.fc(cat)
        # output shape: (batch_size, num_classes)

        return output
# Load the word_to_id mapping
with open('word_to_id.pkl', 'rb') as f:
    word_to_id = pickle.load(f)
# Load the id_to_label mapping
with open('id_to_label.pkl', 'rb') as f:
    id_to_label = pickle.load(f)
with open('model_args.pkl', 'rb') as f:
    model_args = pickle.load(f)

PAD_ID = model_args['pad_idx']
UNK_ID = model_args['unk_idx']
model = TextCNN(
    vocab_size=model_args['vocab_size'],
    embedding_dim=model_args['embedding_dim'],
    num_classes=model_args['num_classes'],
    pad_idx=PAD_ID
)
model.load_state_dict(
    torch.load('textcnn_model.pth', map_location=torch.device('cpu'))
)
model.eval()
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

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

# Token to ID function
def text_to_ids(text):
    preprocessed = preprocess_text(text)
    tokens = word_tokenize(preprocessed)
    return torch.tensor([word_to_id.get(token, UNK_ID) for token in tokens], dtype=torch.long)

# Gradio interface
def classify_text(text):
    kernel_sizes = [3, 4, 5]
    ids = text_to_ids(text)
    MIN_LEN = max(kernel_sizes)
    if len(ids) < MIN_LEN:
        pad = torch.full((MIN_LEN - len(ids),), PAD_ID, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=0)
    inputs = pad_sequence([ids], batch_first=True, padding_value=PAD_ID)
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