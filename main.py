from data_prep import load_doc, load_descriptions, clean_descriptions, save_descriptions
from feature_extraction import extract_features
from model import define_model
from train import train_model
from evaluate import evaluate_model

# Step 1: Prepare data
doc = load_doc('Flickr8k.token.txt')
descriptions = load_descriptions(doc)
clean_descriptions(descriptions)
save_descriptions(descriptions, 'descriptions.txt')

# Step 2: Extract features
features = extract_features('Flickr8k_Dataset/')

# Step 3: Define & train model
vocab_size = 8000  # Example
max_length = 34    # Based on data analysis
model = define_model(vocab_size, max_length)
train_model(model, descriptions, features, tokenizer=None, max_length=max_length, vocab_size=vocab_size, steps=4000, epochs=20)

# Step 4: Evaluate
bleu_score = evaluate_model(model, descriptions, features, tokenizer=None, max_length=max_length)
print(f"BLEU Score: {bleu_score:.2f}")
