from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import os

def extract_features(directory):
    # Global-average pooled 2048-D features
    base = InceptionV3(weights='imagenet', include_top=True)
    model = Model(inputs=base.input, outputs=base.layers[-2].output)  # 'avg_pool' layer (2048)
    features = {}
    for img_name in os.listdir(directory):
        path = os.path.join(directory, img_name)
        img = load_img(path, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        vec = model.predict(x, verbose=0)  # shape (1, 2048)
        features[os.path.splitext(img_name)[0]] = vec
    return features
