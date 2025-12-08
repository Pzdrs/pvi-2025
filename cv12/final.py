import os
import time

import cv2
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

# https://huggingface.co/timm/resnet10t.c3_in1k
MODEL = "resnet10t"
DATA_DIR = os.path.join("data")
LEARNING_RATE = 0.001
EPOCHS = 10
DEV = "mps" if torch.mps.is_available() else "cpu"


def get_dataloaders(data_dir):
    data_transforms = {
        "train": transforms.Compose(
            [
                # model expects 176x176 training images
                transforms.RandomResizedCrop(176),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                # model expects 224x224 validation images
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
        for x in ["train", "valid"]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=100,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
        )
        for x in ["train", "valid"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


# https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train_model(
    dataloaders,
    model,
    criterion,
    optimizer,
    scheduler,
    dataset_sizes,
):
    since = time.time()

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}/{EPOCHS - 1}")
        print("-" * 10)

        for phase in ["train", "valid"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEV)
                labels = labels.to(DEV)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        print()

    time_elapsed = time.time() - since
    print(f"Elapsed time {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    return model


def main():
    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_DIR)
    print(dataset_sizes)

    model_ft = timm.create_model(MODEL, pretrained=True, num_classes=2)

    model_ft = model_ft.to(DEV)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(
        dataloaders,
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        dataset_sizes,
    )

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    child_img = cv2.cvtColor(cv2.imread("data/test/child.jpg"), cv2.COLOR_BGR2RGB)
    adult_img = cv2.cvtColor(cv2.imread("data/test/adult.png"), cv2.COLOR_BGR2RGB)
     
    child_img_t = transform(child_img)
    child_img_t = child_img_t.unsqueeze(0).to(DEV)
    
    audlt_img_t = transform(adult_img)
    audlt_img_t = audlt_img_t.unsqueeze(0).to(DEV)

    with torch.no_grad():
        outputs_child = model_ft(child_img_t)
        probs_child = torch.softmax(outputs_child, dim=1)
        pred_child = torch.argmax(probs_child, dim=1).item()
        
        outputs_adult = model_ft(audlt_img_t)
        probs_adult = torch.softmax(outputs_adult, dim=1)
        pred_adult = torch.argmax(probs_adult, dim=1).item()

    print("Adult Class probabilities:", probs_adult.cpu().numpy())
    print("Predicted class index:", pred_adult)
    print("Predicted class name:", class_names[pred_adult])
    
    print("Child Class probabilities:", probs_child.cpu().numpy())
    print("Predicted class index:", pred_child)
    print("Predicted class name:", class_names[pred_child])


if __name__ == "__main__":
    main()
