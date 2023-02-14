import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    # 线性的beta表
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_x_start_from_image(fp: str | bytes, image_size: int = 128):
    image = Image.open(fp)
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),  # turn into Numpy array of shape HWC, divide by 255
        Lambda(lambda t: (t * 2) - 1),  # turn to (-1,1) from (0, 1)
    ])
    x_start = transform(image).unsqueeze(0)  # add the batch axis
    return x_start


def reverse_to_image_from_x_t(x_t: torch.Tensor):
    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])
    x_reverse = reverse_transform(x_t.squeeze())
    return x_reverse


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


beta_methods = {
    "cosine": cosine_beta_schedule,
    "liner": linear_beta_schedule,
    "quadratic": quadratic_beta_schedule,
    "sigmoid": sigmoid_beta_schedule,
}

if __name__ == '__main__':
    image = Image.open("../t.jpg")
    image.show()

    x_start = get_x_start_from_image("../t.jpg")
    print(x_start.shape)

    reverse_image = reverse_to_image_from_x_t(x_start)
    reverse_image.show()
