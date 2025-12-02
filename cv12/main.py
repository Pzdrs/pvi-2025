import time
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# https://huggingface.co/timm/resnet10t.c3_in1k
MODEL_NAME = "resnet10t"
NUM_CLASSES = 2
BATCH_SIZE = 100
LR = 0.001
EPOCHS = 10
DEV = "mps" if torch.mps.is_available() else "cpu"


def main():
    train_tfms = transforms.Compose(
        [
            # model expects 176x176 training images
            transforms.RandomResizedCrop(176, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    test_tfms = transforms.Compose(
        [
            # model expects 224x224 validation images
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    train_ds = datasets.ImageFolder("data/train", transform=train_tfms)
    valid_ds = datasets.ImageFolder("data/valid", transform=test_tfms)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"Training dataset size: {len(train_ds)}, Validation dataset size: {len(valid_ds)}"
    )

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(
        DEV
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        start_time = time.perf_counter()

        model.train()
        total_loss = 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEV), labels.to(DEV)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        epoch_time = time.perf_counter() - start_time

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"Loss: {total_loss / len(train_dl):.4f} - "
            f"Time: {epoch_time:.2f}s"
        )

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in valid_dl:
            imgs, labels = imgs.to(DEV), labels.to(DEV)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Validation accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()
