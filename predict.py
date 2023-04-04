import argparse

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np

from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', default='./test_images/hard-leaved_pocket_orchid.jpg')
    parser.add_argument('model', default='model.h5')
    parser.add_argument('--category_names', dest='category_names', default='label_map.json')
    parser.add_argument('--top_k', dest='top_k', default='3')
    return parser.parse_args()

def process_image(image):
    image = tf.cast(image,tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()

def load_class_names(filepath):
    with open(filepath, 'r') as f:
        class_names = json.load(f)
    return class_names

def load_model(filepath):
    reloaded_model = tf.keras.models.load_model(filepath, custom_objects={'KerasLayer':hub.KerasLayer})
    return reloaded_model

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    processed_image = process_image(np.asarray(image))
    processed_image = np.expand_dims(processed_image,axis=0)

    pred = model.predict(processed_image)
    
    arg_sort = list(reversed(np.argsort(pred)))
    probs = pred[0][arg_sort[0]]
    classes = arg_sort[0]
    
    return probs, classes

def main():
    args = parse_args()
    model = load_model(args.model)
    class_names = load_class_names(args.category_names)
    
    img_path = args.image
    print('Image Path: ' + img_path)
    
    probs, classes = predict(img_path, model, int(args.top_k))
    labels = [class_names[str(x+1)] for x in classes]
    probability = probs
    
    i=0 # this prints out top k classes and probs as according to user
    for i in range(int(args.top_k)):
        print(f"{labels[i]} (probability: {probability[i]})")

if __name__ == "__main__":
    main()