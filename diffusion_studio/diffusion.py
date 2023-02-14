from diffusion_studio.utils import beta_methods, extract
import torch


class Diffusion(object):

    def __init__(self, timesteps=300, m="cosine"):
        self.timesteps = timesteps
        self.betas = beta_methods[m](timesteps)
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    def forward_add_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        if noise is None:
            noise = torch.randn_like(x_start)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_noicy_images(self, x_start: torch.Tensor, t: torch.Tensor, noise=None):
        x_ts = self.forward_add_noise(x_start, t, noise)
        noicy_images = [reverse_to_image_from_x_t(x_t) for x_t in x_ts]
        return noicy_images


if __name__ == '__main__':
    from diffusion_studio.utils import get_x_start_from_image, reverse_to_image_from_x_t

    diff = Diffusion()
    x_start = get_x_start_from_image("../test.jpg")
    t = [20, 40, 80, 160, 299]
    x_ts = diff.get_noicy_images(x_start, torch.tensor(t))
    i_0 = reverse_to_image_from_x_t(x_start)
    i_0.save("start.jpg")
    for i, x_t in zip(t, x_ts):
        # i_t.show()
        x_t.save(f"{i}.jpg")
