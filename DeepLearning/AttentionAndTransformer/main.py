# Nyoba dari AI

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dataset
texts = [
    "i love this movie",
    "this movie is amazing",
    "fantastic film",
    "this is wonderful",

    "i hate this movie",
    "this movie is terrible",
    "worst movie ever",
    "this is awful"
]

# 1: Positive, 0: Negative
labels = [1, 1, 1, 1, 0, 0, 0, 0]  

# Vectorizer
max_tokens = 1000
sequence_length = 6

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length
)
vectorizer.adapt(texts)

# Positional Encoding
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

# Self Attention
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = tf.keras.layers.Dense(embed_dim)
        self.Wk = tf.keras.layers.Dense(embed_dim)
        self.Wv = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        score = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(self.embed_dim, tf.float32)
        score = score / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(score, axis=-1)
        output = tf.matmul(attention_weights, V)
        return output, attention_weights

# Multi Head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim harus bisa dibagi habis oleh num_heads"

        self.Wq = tf.keras.layers.Dense(embed_dim)
        self.Wk = tf.keras.layers.Dense(embed_dim)
        self.Wv = tf.keras.layers.Dense(embed_dim)
        self.Wo = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]

        Q = self.split_heads(self.Wq(x), batch_size)
        K = self.split_heads(self.Wk(x), batch_size)
        V = self.split_heads(self.Wv(x), batch_size)

        score = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(self.head_dim, tf.float32)
        score = score / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(score, axis=-1)
        self.last_attention = attention_weights
        output = tf.matmul(attention_weights, V)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.embed_dim))

        final_output = self.Wo(concat_attention)
        return final_output, attention_weights

# Feed Forward
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ff_dim, activation="relu")
        self.dense2 = tf.keras.layers.Dense(embed_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

# Encoder
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.ffn = FeedForward(embed_dim, ff_dim)
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn_out, _ = self.mha(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# Transformer Classifier
class TransformerClassifier(tf.keras.Model):
    def __init__(self, sequence_length, vocab_size, embed_dim, num_heads, ff_dim, num_classes):
        super().__init__()
        self.vectorizer = vectorizer
        self.pos_embedding = PositionalEmbedding(sequence_length, vocab_size, embed_dim)
        self.encoder = TransformerEncoder(embed_dim, num_heads, ff_dim)
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.fc1 = tf.keras.layers.Dense(32, activation="relu")
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.vectorizer(inputs)
        x = self.pos_embedding(x)
        x = self.encoder(x)
        x = self.pooling(x)
        x = self.fc1(x)
        outputs = self.classifier(x)
        return outputs

# Instansiasi Model
embed_dim = 32
num_heads = 4
ff_dim = 64
num_classes = 2

model = TransformerClassifier(
    sequence_length=sequence_length,
    vocab_size=max_tokens,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_classes=num_classes
)

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
x_train = tf.constant(texts, dtype=tf.string)
y_train = tf.constant(labels, dtype=tf.int32)

print("--- Memulai Training ---")
history = model.fit(x_train, y_train, epochs=40, verbose=0)
print("Training Selesai!")

# Evaluate
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print(f"Loss Training    : {loss:.4f}")
print(f"Akurasi Training : {acc * 100:.2f}%\n")

# Predict
test_samples = tf.constant([
    "this movie is fantastic",
    "this movie is terrible",
    "i love this movie",
    "worst movie ever"
], dtype=tf.string)

predictions = model(test_samples)
attention = model.encoder.mha.last_attention
classes = ["Negative", "Positive"]

print("--- Hasil Prediksi ---")
for kalimat, probabilitas in zip(test_samples.numpy(), predictions):
    idx = tf.argmax(probabilitas).numpy()
    print(f"Kalimat   : {kalimat.decode()}")
    print(f"Prediksi  : {classes[idx]}")
    print(f"Confidence: {probabilitas[idx] * 100:.2f}%")
    print("-" * 40)
