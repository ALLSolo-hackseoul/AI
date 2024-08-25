import os
import json
import pandas as pd
import tqdm


file_info = pd.read_csv("/workspace/data/img.csv")
annot = pd.read_csv("/workspace/data/annot.csv")

with tqdm.tqdm(file_info["id"]) as t:
    for id in t:
        string = ""
        f = open(f"/workspace/data/labels/train/{id}.txt", "wt")
        coors =  annot[annot.image_id == id]["bbox"]
        width = int(file_info[file_info.id == id]["width"].item())
        height = int(file_info[file_info.id == id]["height"].item())
        for coor in coors:
            coor = coor[1:-1]
            coor = coor.split(" ")
            coor = [x.replace(",", "") for x in coor]
            string += f"0 {float(coor[0]) / width} {float(coor[1]) / height} {float(coor[2]) / width} {float(coor[3]) / height}\n"
        f.write(string)
        f.close()