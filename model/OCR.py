import os
import pandas as pd
import numpy as np


file_list = os.listdir("data\\OCR\\train_val_images\\train_images")
annot = pd.read_csv("data\\OCR\\annot.csv")

for file in file_list:
    annot
