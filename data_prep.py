import string
import os
import numpy as np

def load_doc(filename):
    with open(filename, 'r') as f:
        text = f.read()
    return text

def load_descriptions(doc):
    mapping = {}
    for line in doc.strip().split("\n"):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        mapping.setdefault(image_id, []).append(image_desc)
    return mapping

def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i, desc in enumerate(desc_list):
            desc = desc.lower().translate(table)
            desc = ' '.join([word for word in desc.split() if len(word) > 1 and word.isalpha()])
            descriptions[key][i] = desc

def save_descriptions(descriptions, filename):
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(f"{key} {desc}")
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))
