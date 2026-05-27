import cv2
import torch 
from model import CNN
import numpy as np 
import requests 
import cv2
import numpy as np
import requests
import os


def process_image(path_or_url):
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        response = requests.get(path_or_url)
        if response.status_code != 200:
            return None
        image_array = np.asarray(
            bytearray(response.content),
            dtype=np.uint8
        )
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    else:
        image = cv2.imread(path_or_url)

    if image is None:
        return None

    image = cv2.resize(image, dsize=(224,224))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return np.array(image)
    
model=CNN()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "best_model.pth"))) 

def predict(image_path):
    image =process_image(image_path)
    if image is not None:
        image=torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        model.eval()
        device=torch.device("cpu")
        with torch.no_grad():
            image=image.float().to(device)
            image=image.unsqueeze(0)
            pred=model(image)
            output=torch.round(pred)
            if output.item()==0:
                return "Bird"
            else:
                return "Drone"
            
            
if __name__=="__main__":
    img_address="https://thumbs.dreamstime.com/b/bird-perched-tree-branch-30-36-jpg-5136761.jpg"
    prediction=predict(img_address)
    print(f"Predicted class : {prediction}")
    
        