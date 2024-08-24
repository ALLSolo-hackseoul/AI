import torch
import torchvision
import torch.utils
import torch.utils.data
import tqdm
from model.autoencoder import AutoEncoder


EPOCHS = 10
lr = 5e-4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def main():
    model = AutoEncoder()

    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([128, 128]),
            torchvision.transforms.ToTensor()
        ])
    train_dataloader = torchvision.datasets.ImageFolder(root="data", transform=transform)

    model.train()
    model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    

    for epoch in range(EPOCHS):
        with tqdm.tqdm(train_dataloader, unit="batch") as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                optimizer.zero_grad()
                    
                output, hidden = model(x)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()
                
                tepoch.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), "model.pt")
    
if __name__ == "__main__":
    main()
