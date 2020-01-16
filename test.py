
import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from train import read_image

if __name__ == "__main__":
    path = "Dog Breed Identification/"
    test_path = os.path.join(path, "test/*")
    labels_path = os.path.join(path, "labels.csv")

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    print("Number of Breeds: ", len(breed))
    id2breed = {i:name for i, name in enumerate(breed)}

    images = glob(test_path)[:10]

    model = tf.keras.models.load_model("model.h5")

    for i, path in tqdm(enumerate(images)):
        image = read_image(path, 224)           ## (224, 224, 3)
        image = np.expand_dims(image, axis=0)   ## (1, 224, 224, 3)
        pred = model.predict(image)[0]
        idx = np.argmax(pred)
        breed_name = id2breed[idx]

        ## Original_image
        ori_image = cv2.imread(path, cv2.IMREAD_COLOR)

        ## Save
        save_image = cv2.putText(ori_image, breed_name, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(f"save/test_{i+1}.png", save_image)
