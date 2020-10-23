import torch
from torchvision import transforms

model_path = "wagglecloud_deeplab_300.pth"
image_size = 300


def preprocess(image):
    f = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return f(image)


def postprocess(output):
    output = output[0]
    output_predictions = output.argmax(0)

    scores = output_predictions.cpu().numpy().reshape(-1)
    cloud = 0
    for i in scores:
        if i == 1:
            cloud += 1
    return cloud / len(scores)
