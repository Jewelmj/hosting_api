import streamlit as st
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer, T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import re
import string

# -------------------- CLEANING FUNCTION --------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(rf"[{string.punctuation}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------- LOAD ROBERTA TOKENIZER --------------------
tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
max_len = 200

def encode_roberta(sentence):
    tokens = tokenizer_roberta.encode(sentence, truncation=True, max_length=max_len, add_special_tokens=True)
    return torch.tensor(tokens, dtype=torch.long)

# -------------------- BiLSTM MODEL --------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, output_dim=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer_roberta.pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        out = self.dropout(h)
        return self.fc(out)

# -------------------- LOAD MODELS --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer_roberta.vocab_size

bilstm_model = BiLSTMClassifier(vocab_size).to(device)
bilstm_model.load_state_dict(torch.load("lstm_model.pt", map_location=device))
bilstm_model.eval()

t5_tokenizer = T5Tokenizer.from_pretrained("enco_deco_tokenizer_t5small")
t5_model = T5ForConditionalGeneration.from_pretrained("enco_deco_model_t5small").to(device)
t5_model.eval()

# -------------------- STREAMLIT UI --------------------
st.title("💬 Sales Response Improver")

user_input = st.text_area("Enter a salesperson's response:", height=150)

if st.button("Analyze & Improve"):
    with st.spinner("Analyzing..."):
        cleaned_input = clean_text(user_input)
        encoded_input = pad_sequence([encode_roberta(cleaned_input)], batch_first=True,
                                     padding_value=tokenizer_roberta.pad_token_id).to(device)

        # Get sentiment rating
        with torch.no_grad():
            logits = bilstm_model(encoded_input)
            predicted_rating = logits.argmax(dim=1).item() + 1  # 1–5 scale

        st.write(f"⭐ **Predicted Rating:** {predicted_rating}/5")

        # Generate improved response if rating is bad
        if predicted_rating <= 3:
            t5_inputs = t5_tokenizer(cleaned_input, return_tensors="pt", padding="max_length",
                                     truncation=True, max_length=200).to(device)

            output_ids = t5_model.generate(
                input_ids=t5_inputs["input_ids"],
                attention_mask=t5_inputs["attention_mask"],
                max_length=200,
                num_beams=4,
                early_stopping=True
            )

            improved = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Get new rating
            improved_cleaned = clean_text(improved)
            encoded_improved = pad_sequence([encode_roberta(improved_cleaned)], batch_first=True,
                                            padding_value=tokenizer_roberta.pad_token_id).to(device)

            with torch.no_grad():
                improved_rating = bilstm_model(encoded_improved).argmax(dim=1).item() + 1

            st.success("✅ Improved Response:")
            st.text_area("Improved Response", improved, height=150)
            st.write(f"✨ **Improved Rating:** {improved_rating}/5")
        else:
            st.info("This response is already good and doesn't need improvement.")

