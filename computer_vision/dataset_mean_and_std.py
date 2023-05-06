import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm


def get_mean_and_std_of_rgb_values(data_dir, ext="jpg"):
    data_dir = Path(data_dir)

    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for img_path in tqdm(list(data_dir.glob(f"""**/*.{ext}"""))):
        pil_img = Image.open(img_path)
        tensor = T.ToTensor()(pil_img)
        
        sum_rgb += tensor.sum(dim=(1, 2))
        sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
        _, h, w = tensor.shape
        sum_resol += h * w
    mean = torch.round(sum_rgb / sum_resol, decimals=3)
    std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
    return mean, std


if __name__ == "__main__":
    mean, std = get_mean_and_std_of_rgb_values("/Users/jongbeomkim/Documents/datasets/VOCdevkit/VOC2012/JPEGImages")
    print(mean, std)
