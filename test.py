import torch
import torchvision
import torch.utils
import torch.utils.data
from model.autoencoder import AutoEncoder
from transformers import get_scheduler
from transformers import ViTImageProcessor, ViTForImageClassification

def main():
    model = AutoEncoder()
    model.load_state_dict(torch.load("model.pt"))

    resize = torchvision.transforms.Resize([128, 128])

    data1 = torchvision.io.read_image("src.png").to(torch.float)
    data1_clone = torchvision.io.read_image("target.png").to(torch.float)

    data1 = resize(data1)
    data1_clone = resize(data1_clone)

    data1 /= 255
    
    data1_clone /= 255

    _, output1 = model(data1)
    _, output1_clone = model(data1_clone)

    output1 = torch.flatten(output1)
    output1_clone = torch.flatten(output1_clone)
    

    similarity = torch.nn.functional.cosine_similarity(output1,output1_clone, dim=0)

    print("similarity :" ,similarity.item())

if __name__ == "__main__":
    main()
