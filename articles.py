import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GRU
from tensorflow.keras.models import Sequential
import numpy as np
import os

# Read the data
text = open('articles.txt', 'rb').read().decode(encoding='utf-8')

# Create a unique sorted list of all characters in the text
vocab = sorted(set(text))

# Create a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert the text to a sequence of character indices
text_as_int = np.array([char2idx[c] for c in text])

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

# Define the model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = Sequential([
    Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
    Dense(vocab_size)
  ])
  return model

# Define the loss function
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Function to build and train the model
def train_model(model):
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    EPOCHS=5
    history = model.fit(dataset, epochs=EPOCHS)
    return model

# Function to save the model
def save_model(model):
    model.save_weights('article_model.ckpt')

# Function to load the model
def load_model(batch_size):
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=batch_size)
    model.load_weights('article_model.ckpt')
    return model

# Function to generate text
def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in "@START" + start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = ''
    temperature = 1.0
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated += idx2char[predicted_id]
        if text_generated.endswith('@END'):
            text_generated = text_generated[:-len('@END')]
            break
    return (start_string + ''.join(text_generated))

trainable_model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

# Interactive session
while True:
    command = input("Enter a command (train/load/generate/quit): ")
    if command == 'train':
        trainable_model = train_model(trainable_model)
        save_model(trainable_model)
        generation_model = load_model(1)
    elif command == 'load':
        if os.path.exists('article_model.ckpt.index'):
            trainable_model = load_model(BATCH_SIZE)
            generation_model = load_model(1)
        else:
            print("No saved model found.")
    elif command == 'generate':
        if 'generation_model' in locals():
            start_string = input("Enter a starting string: ")
            print(generate_text(generation_model, start_string=start_string))
        else:
            print("No model available. Train or load a model first.")
    elif command == 'quit':
        break
    else:
        print("Invalid command.")
