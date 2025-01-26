import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define constants
TEST_DIR = "./Dataset/Images"
TARGET_SIZE = (150, 150) 
CLASS_NAMES = ["Brain Tumor", "Healthy"] 

# Function to predict the class of an image
def predict_image(model, img_path, target_size):
    img = load_img(img_path, target_size=target_size) 
    img_array = img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    prediction = model.predict(img_array)
    predicted_class = int(prediction[0] > 0.5) 
    return predicted_class, prediction[0][0]

# Iterate through test folders
correct_predictions = 0
total_images = 0
model = keras.models.load_model('./Trained_Models/Sequential_13_0.999.keras')

for true_class, folder in enumerate(CLASS_NAMES):
    folder_path = os.path.join(TEST_DIR, folder)
    for filename in os.listdir(folder_path):
        total_images += 1
        file_path = os.path.join(folder_path, filename)
        predicted_class, confidence = predict_image(model, file_path, TARGET_SIZE)

        is_correct = (predicted_class == true_class)
        correct_predictions += is_correct

        print(f"File: {filename}")
        print(f"True Class: {CLASS_NAMES[true_class]}, Predicted Class: {CLASS_NAMES[predicted_class]}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Correct: {is_correct}\n")

# Summary of results
accuracy = correct_predictions / total_images
print(f"Total Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2%}")
