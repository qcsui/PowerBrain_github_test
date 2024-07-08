import requests
from PIL import Image
import json

def test_predict():
    url = "http://127.0.0.1:5000/predict"
    img_path = "static/img_test/test_image.jpg"
    
    # Open the image file in binary mode
    with open(img_path, 'rb') as img:
        files = {"file": img}
        response = requests.post(url, files=files)
        print(f"response.json({response.json()})")

if __name__ == "__main__":
    test_predict()