import numpy as np
import tensorflow as tf

# Example data
text = "a quick brown fox jumps over the lazy dog"
words = text.split()
word_to_id = {word: idx for idx, word in enumerate(set(words))}
id_to_word = {idx: word for word, idx in word_to_id.items()}
vocab_size = len(word_to_id)

# Convert words to IDs
word_ids = [word_to_id[word] for word in words]

# Create input and output sequences
seq_length = 3
x_data = []
y_data = []
for i in range(len(word_ids) - seq_length):
    x_data.append(word_ids[i:i + seq_length])
    y_data.append(word_ids[i + seq_length])  # Next word after the sequence

# Convert to numpy arrays
x_data = np.array(x_data)
y_data = np.array(y_data)

# Parameters for the network
num_epochs = 10
batch_size = 2  # Adjust batch size if needed
learning_rate = 0.01
hidden_size = 10

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, hidden_size, input_length=seq_length),
    tf.keras.layers.LSTM(hidden_size),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate))

# Train the model
model.fit(x_data, y_data, epochs=num_epochs, batch_size=batch_size, verbose=2)


# Function to generate text
def generate_text(seed_text, num_words):
    seed_ids = [word_to_id[word] for word in seed_text.split()]
    generated_ids = seed_ids.copy()

    for _ in range(num_words):
        padded_seed = np.array([seed_ids[-seq_length:]])  # Take only the last seq_length words
        predicted_probs = model.predict(padded_seed, verbose=0)[0]
        predicted_id = np.argmax(predicted_probs)
        generated_ids.append(predicted_id)
        seed_ids.append(predicted_id)

    generated_text = ' '.join([id_to_word[idx] for idx in generated_ids])
    return generated_text


# Generate sample text
generated_text = generate_text("quick brown", 5)
print("Generated Text: ", generated_text)
