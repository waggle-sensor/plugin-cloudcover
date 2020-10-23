import torch
from torchvision import transforms

model_path = "wagglecloud_unet_300.pth"
image_size = 300


def preprocess(image):
    f = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    return f(image)


def postprocess(output, threshold=0.5):
    probs = torch.sigmoid(output)
    probs = probs.squeeze(0)
    scores = probs.detach().cpu().numpy().reshape(-1)

    maxs = max(scores)
    mins = min(scores)
    scores = [(i-mins)/(maxs-mins) for i in scores]

    for i in range(len(scores)):
        if scores[i] > threshold:
            scores[i] = True
        else:
            scores[i] = False

    cloud = 0
    for i in scores:
        if i == 1:
            cloud += 1
    return cloud / len(scores)
