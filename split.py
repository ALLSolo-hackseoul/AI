import os
import random
import shutil


img_list = os.listdir("/workspace/data/train_val_images/train_images")

for x in img_list:
    id = x.replace(".jpg", "")
    if not os.path.exists(f"/workspace/data/labels/train/{id}.txt"):
        try:
            os.remove(x)
        except:
            continue
