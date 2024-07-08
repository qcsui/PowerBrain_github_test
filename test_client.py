# test app.py's predict function
import requests
import json
import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms
# Server code: 
# ```python
# # A flask app for calculate MNIST dataset using a trained model
# # input is a image (unknown size) and output is a prediction. 
# import os
# import numpy as np
# from flask import Flask, request, jsonify
# import torch
# import torchvision
# from torchvision import transforms
# from PIL import Image
# from model import CNN


# app = Flask(__name__)

# # Load model
# model = CNN()
# model.load_state_dict(torch.load("cnn.pth"))
# model.eval()

# # Load class names
# class_names = [
#     "0", "1", "2", "3", "4",
#     "5", "6", "7", "8", "9"
# ]

# # Preprocess image
# def preprocess_image(image):
#     image = image.convert("L")
#     transform = transforms.Compose([
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     image = transform(image).unsqueeze(0)
#     return image

# @app.route("/predict", methods=["POST"])
# def predict():
#     if request.method == "POST":
#         # Check if the post request has the file part
#         if "file" not in request.files:
#             return jsonify({"error": "no file part"})
#         file = request.files["file"]
#         if file.filename == "":
#             return jsonify({"error": "no selected file"})
#         if file:
#             try:
#                 image = Image.open(file)
#                 image = preprocess_image(image)
#                 with torch.no_grad():
#                     predictions = model(image)
#                     _, predicted = torch.max(predictions, 1)
#                     result = {"prediction": class_names[predicted[0].item()]}
#                     return jsonify(result)
#             except Exception as e:
#                 return jsonify({"error": str(e)})
#     return jsonify({"error": "invalid request"})
# if __name__ == "__main__":
#     app.run()
# ```

# Test the server
# Load the image
img = Image.open("test_image.jpg")
# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
img = transform(img).unsqueeze(0)
# Send the image to the server
url = "http://127.0.0.1:5000/predict"
files = {"file": open("test_image.jpg", "rb")}
r = requests.post(url, files=files)
print(r.text)

