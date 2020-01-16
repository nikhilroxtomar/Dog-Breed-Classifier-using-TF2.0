
import os
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split

## Read Image
def read_image(path, size):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (size, size))
    image = image / 255.0
    image = image.astype(np.float32)
    return image

if __name__ == "__main__":
    path = "../../../ml_dataset/Dog Breed Identification/"
    train_path = os.path.join(path, "train/*")
    test_path = os.path.join(path, "test/*")
    labels_path = os.path.join(path, "labels.csv")

    test_images = glob(test_path)

    labels_df = pd.read_csv(labels_path)
    breed = labels_df["breed"].unique()
    breed2id = {name: i for i, name in enumerate(breed)}
    id2breed = {i: name for i, name in enumerate(breed)}

    ids = glob(train_path)
    labels = []
    for image_id in ids:
         image_id = image_id.split("/")[-1].split(".")[0]
         breed_idx = breed2id[list(labels_df[labels_df.id == image_id]["breed"])[0]]
         labels.append(breed_idx)

    ids = ids[:5000]
    labels = labels[:5000]

    train_x, valid_x = train_test_split(ids, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(labels, test_size=0.2, random_state=42)

    ## Model
    model = tf.keras.models.load_model("model.h5")

    ## Valid Images
    for i, path in tqdm(enumerate(valid_x[:20])):
        image = read_image(path, 224)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)[0]
        label_idx = np.argmax(pred)
        breed_name = id2breed[label_idx]

        ##
        ori_breed = id2breed[valid_y[i]]

        ## Original image
        ori_image = cv2.imread(path, cv2.IMREAD_COLOR)

        ## Original Breed
        ori_image = cv2.putText(ori_image, breed_name, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        ## Predicted Breed
        ori_image = cv2.putText(ori_image, ori_breed, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(f"save/valid_{i}.png", ori_image)

    # ## Test Images
    # for i, path in tqdm(enumerate(test_images[:10])):
    #     image = read_image(path, 224)
    #     image = np.expand_dims(image, axis=0)
    #     pred = model.predict(image)[0]
    #     label_idx = np.argmax(pred)
    #     breed_name = id2breed[label_idx]
    #
    #     ## Original image
    #     ori_image = cv2.imread(path, cv2.IMREAD_COLOR)
    #     save_image = cv2.putText(ori_image, breed_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    #     cv2.imwrite(f"save/{i}.png", save_image)
