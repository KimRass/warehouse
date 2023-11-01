# Reference
    # https://github.com/facebookresearch/deit/blob/main/models.py

import torch
import torch.nn as nn

from vit import PatchEmbedding, TransformerEncoder


class DeiT(nn.Module):
    # embedding #heads #layers #params training throughput
    # DeiT-Ti: `hidden_dim=192, n_heads=3, n_layers=12`
    # DeiT-S: `hidden_dim=384, n_heads=6, n_layers=12`
    # DeiT-B: `hidden_dim=768, n_heads=12, n_layers=12`
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        hidden_dim=192,
        n_heads=3,
        n_layers=12,
        n_classes=1000,
        training=False
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.training = training

        assert img_size % patch_size == 0,\
            "`img_size` must be divisible by `patch_size`!"

        grid_size = img_size // patch_size
        n_patches = grid_size ** 2

        self.patch_embed = PatchEmbedding(patch_size=patch_size, hidden_dim=hidden_dim)

        self.cls_token = nn.Parameter(torch.randn((1, 1, hidden_dim)))
        self.distil_token = nn.Parameter(torch.randn((1, 1, hidden_dim)))
        self.pos_embed = nn.Parameter(torch.randn((1, n_patches + 2, hidden_dim)))
        self.dropout = nn.Dropout(0.5)

        self.tf_enc = TransformerEncoder(n_layers=n_layers, hidden_dim=hidden_dim, n_heads=n_heads)

        self.ln = nn.LayerNorm(hidden_dim)

        self.cls_mlp = nn.Linear(hidden_dim, n_classes)
        self.distil_mlp = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        b, _, _, _ = x.shape

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.repeat(b, 1, 1), self.cls_token.repeat(b, 1, 1), x), dim=1)
        x += self.pos_embed
        x = self.dropout(x)

        x = self.tf_enc(x)

        x1, x2 = x[:, 0], x[:, 1]
        x1, x2 = self.ln(x1), self.ln(x2)

        x1, x2 = self.cls_mlp(x1), self.distil_mlp(x2)
        if self.training:
            return x1, x2
        else:
            return (x1 + x2) / 2


if __name__ == "__main__":
    image = torch.randn((4, 3, 224, 224))
    deit = DeiT(training=True)
    cls_output, distil_output = deit(image)
    print(cls_output.shape, distil_output.shape)
