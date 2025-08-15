import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    while True:
        for key, desc_list in descriptions.items():
            photo = photos[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield ([photo, in_seq], out_seq)

def train_model(model, train_descriptions, train_features, tokenizer, max_length, vocab_size, steps, epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
    model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)
