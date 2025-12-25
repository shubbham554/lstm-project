# train_lstm.py
# ----------------------------
# A minimal LSTM next-word predictor trainer
# ----------------------------

import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# ----------------------------
# 1. Sample text corpus
# You can expand this with your own text dataset later
# ----------------------------
data = """
artificial intelligence is the future
machine learning is part of artificial intelligence
deep learning uses neural networks
neural networks learn from data
data science is growing fast
ai and ml are transforming the world
"""

# ----------------------------
# 2. Tokenize text
# ----------------------------
corpus = [line.strip() for line in data.lower().split("\n") if line.strip()]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# ----------------------------
# 3. Create input sequences for next-word prediction
# ----------------------------
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X, labels = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(labels, num_classes=total_words)

# ----------------------------
# 4. Build the LSTM model
# ----------------------------
model = Sequential([
    Embedding(total_words, 50, input_length=max_sequence_len-1),
    LSTM(100),
    Dense(total_words, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ----------------------------
# 5. Train
# ----------------------------
print("Training LSTM model...")
model.fit(X, y, epochs=200, verbose=0)
print("✅ Training complete!")

# ----------------------------
# 6. Save model and tokenizer
# ----------------------------
model.save("model/lstm_model.h5")
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model and tokenizer saved successfully!")
print(f"Total words in vocab: {total_words}")
print(f"Max sequence length: {max_sequence_len}")
