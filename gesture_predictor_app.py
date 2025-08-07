import torch
import torch.nn as nn
import tkinter as tk
from tkinter import messagebox

# === Vocabulary and tokenizer ===
vocab = [
    "i", "want", "to", "drink", "eat", "stop", "help", "need", "can", "you",
    "me", "please", "<mask>", "<pad>", "pain", "toilet", "hello",
    "thank", "thank you", "yes", "no", "feel", "say", "the", "wants", "hi"
]
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(vocab)
max_len = 8
d_model = 64

def tokenize(sentence):
    words = sentence.strip().lower().split()
    return [word2idx.get(word, word2idx["<pad>"]) for word in words]

def encode_and_pad(sentence, max_len):
    ids = tokenize(sentence)
    if len(ids) < max_len:
        ids += [word2idx["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]

def get_positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    return angle_rads

# === Model definition ===
class ConfidenceTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.conf_proj = nn.Linear(1, d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, conf):
        seq_len = x.size(1)
        embed = self.embedding(x)
        pos = get_positional_encoding(seq_len, embed.size(-1)).to(x.device)
        embed = embed + pos
        attn_output, _ = self.attn(embed, embed, embed)
        x = self.norm1(embed + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        conf_vec = self.conf_proj(conf).unsqueeze(1)
        x = x + conf_vec
        return self.fc_out(x)

# === Load the trained model ===
model = ConfidenceTransformer(vocab_size, d_model)
model.load_state_dict(torch.load("confidence_transformer1.pth", map_location=torch.device('cpu')))
model.eval()

# === GUI ===
def predict():
    sentence = sentence_entry.get().strip()
    confidence_str = conf_entry.get().strip()

    if not sentence or "<MASK>" not in sentence.upper():
        messagebox.showerror("Error", "Please Enter A Sentence with '<MASK>'.")
        return

    try:
        confidence = float(confidence_str)
        if not 0 <= confidence <= 1:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Confidence Must Be A Number Between 0 and 1.")
        return

    # Tokenize, pad, find <MASK> position
    tokens = sentence.lower().split()
    try:
        mask_pos = tokens.index("<mask>")
    except ValueError:
        messagebox.showerror("Error", "No <MASK> token found.")
        return

    x_ids = encode_and_pad(sentence, max_len)
    x_tensor = torch.tensor([x_ids])
    conf_tensor = torch.tensor([[confidence]], dtype=torch.float32)

    with torch.no_grad():
        logits = model(x_tensor, conf_tensor)
        pred_idx = torch.argmax(logits[:, mask_pos, :], dim=-1).item()
        pred_word = idx2word[pred_idx]
        user_type = "Normal" if confidence >= 0.7 else "Aphasic"
        result_label.config(text=f"Predicted Word: {pred_word}\nUser Type: {user_type}")

# === Build GUI ===
window = tk.Tk()
window.title("Word Prediction with Confidence Transformer (Normal People and Aphasic)")
window.geometry("450x250")
window.resizable(False, False)

tk.Label(window, text="Enter Sentence (use <MASK>):").pack(pady=5)
sentence_entry = tk.Entry(window, width=60)
sentence_entry.pack()

tk.Label(window, text="Confidence (0 to 1):").pack(pady=5)
conf_entry = tk.Entry(window, width=20)
conf_entry.pack()

tk.Button(window, text="Predict", command=predict).pack(pady=10)

result_label = tk.Label(window, text="", font=("Arial", 12), fg="blue")
result_label.pack(pady=10)

window.mainloop()


