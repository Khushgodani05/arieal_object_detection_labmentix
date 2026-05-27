import os 
import cv2
import matplotlib.pyplot as plt
import glob as glob
from concurrent.futures import ThreadPoolExecutor,as_completed
from tqdm import tqdm
import numpy as np 

bird_base_path_train = "../Data/train/bird"
drone_base_path_train = "../Data/train/drone"
bird_base_path_test = "../Data/test/bird"
drone_base_path_test = "../Data/test/drone" 
bird_base_path_val = "../Data/valid/bird"
drone_base_path_val = "../Data/valid/drone"

bird_train_file = os.listdir(bird_base_path_train)
drone_train_file = os.listdir(drone_base_path_train)
bird_test_file=os.listdir(bird_base_path_test)
drone_test_file=os.listdir(drone_base_path_test)
bird_val_file=os.listdir(bird_base_path_val)
drone_val_file=os.listdir(drone_base_path_val)

bird_path = os.path.join(bird_base_path_train, bird_train_file[0])
drone_path = os.path.join(drone_base_path_train, drone_train_file[0])

img_bird = cv2.imread(bird_path)
img_drone = cv2.imread(drone_path)

if img_bird is None:
    print("Bird image not found!")
else:
    img_bird = cv2.resize(img_bird, (224, 224))
    img_bird = cv2.cvtColor(img_bird, cv2.COLOR_BGR2RGB)
    plt.imshow(img_bird)
    plt.show()
    
if img_drone is None:
    print("Drone image not found!")
else:
    img_drone = cv2.resize(img_drone, (224, 224))
    img_drone = cv2.cvtColor(img_drone, cv2.COLOR_BGR2RGB)
    plt.imshow(img_drone)
    plt.show()
    


def process_image(path):
    img=cv2.imread(path)
    try:
        if img is not None:
            img=cv2.resize(img,dsize=(224,224))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            return img
        else:
            print(f"Error: Unable to read image at {path}")
            return None
    except Exception as e:
        print(f"Error processing image at {path}: {e}")
        return None
    
    except cv2.error as e:
        print(f"OpenCV error processing image at {path}: {e}")
        return None
    
def load_images(path):
    images=[]
    file_pattern=os.path.join(path,"*.jpg")
    filenames=glob.glob(file_pattern)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_filename = {executor.submit(process_image, filename): filename for filename in filenames[0: int(1*len(filenames))]}
        for future in tqdm(as_completed(future_to_filename), total=len(filenames)):
            result = future.result()
            if result is not None:
                images.append(result)

    return np.array(images)


bird_train=load_images(bird_base_path_train)
drone_train=load_images(drone_base_path_train)
bird_test=load_images(bird_base_path_test)
drone_test=load_images(drone_base_path_test)
bird_val=load_images(bird_base_path_val)
drone_val=load_images(drone_base_path_val)

birdtrainlabel=np.array([0]*bird_train.shape[0]).reshape(-1,1)
birdtestlabel=np.array([0]*len(bird_test)).reshape(-1,1)
birdvallabel=np.array([0]*len(bird_val)).reshape(-1,1)
dronetrainlabel=np.array([1]*drone_train.shape[0]).reshape(-1,1)
dronetestlabel=np.array([1]*len(drone_test)).reshape(-1,1)
dronevallabel=np.array([1]*len(drone_val)).reshape(-1,1)
# print(f"Bird Train label shape: {birdtrainlabel.shape}\nBird_test label shape : {birdtestlabel.shape}\nBird_val label shape : {birdvallabel.shape}\nDrone_train label shape : {dronetrainlabel.shape}\nDrone_test label shape : {dronetestlabel.shape}\nDrone_val label shape : {dronevallabel.shape}")

x_train=np.concatenate([bird_train, drone_train], axis=0)
y_train=np.concatenate([birdtrainlabel, dronetrainlabel], axis=0)
x_val=np.concatenate([bird_val, drone_val], axis=0)
y_val=np.concatenate([birdvallabel, dronevallabel], axis=0)
x_test=np.concatenate([bird_test, drone_test], axis=0)
y_test=np.concatenate([birdtestlabel, dronetestlabel], axis=0)
# print(f"x_train shape : {x_train.shape}\ny_train shape : {y_train.shape}\nx_val shape : {x_val.shape}\ny_val shape : {y_val.shape}\nx_test shape : {x_test.shape}\ny_test shape : {y_test.shape}")

os.makedirs("../Data/numpy", exist_ok=True)
# np.save("../Data/numpy/bird_train.npy", bird_train)
# np.save("../Data/numpy/drone_train.npy", drone_train)
# np.save("../Data/numpy/bird_test.npy", bird_test)
# np.save("../Data/numpy/drone_test.npy", drone_test)
# np.save("../Data/numpy/bird_val.npy", bird_val)
# np.save("../Data/numpy/drone_val.npy", drone_val)
np.save("../Data/numpy/x_train.npy", x_train)
np.save("../Data/numpy/y_train.npy", y_train)       
np.save("../Data/numpy/x_val.npy", x_val)
np.save("../Data/numpy/y_val.npy", y_val)
np.save("../Data/numpy/x_test.npy", x_test)
np.save("../Data/numpy/y_test.npy", y_test)