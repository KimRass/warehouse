# References
    # https://github.com/openai/CLIP/blob/main/clip/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIP(nn.Module):
    def __init__(self, embed_dim, width, height, patch_size, vocab_size, hidden_size=768):
        super().__init__()

        # ResNet or Vision Transformer
        self.vit = ViT()
        # CBOW or Text Transformer
        self.transformer = Transformer(width=width, height=height, patch_size=patch_size)

        self.vocab_size = vocab_size

        d_model = 512
        self.token_embed = nn.Embedding(vocab_size, d_model)

        self.image_projection = nn.Linear(hidden_size, embed_dim)
        self.text_projection = nn.Linear(hidden_size, embed_dim)

    def encode_image(self, image):
        return self.vit(image)

    def encode_text(self, text):
        x = self.token_embed(text)
        x = self.transformer(text) #(batch_size, seq_len, d_model)
        # ...
        # The activations of the highest layer of the transformer at the `"[EOS]"` token are
        # treated as the feature representation of the text
        # which is layer normalized and then linearly projected into the multi-modal embedding space.


    def forward(self, img, text):
        image_features = self.image_encoder(img) # (n, d_i)
        text_features = self.text_encoder(text) # (n, d_t)

        ### Normalize
        image_features = F.normalize(input=image_features, p=2, dim=1)
        text_features = F.normalize(input=text_features, p=2, dim=1)
    

    batch_size = 8
    seq_len = 76
    d_model = 512
    vocab_size = 1_000
    text = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    token_embed = nn.Embedding(vocab_size, d_model)
    x = token_embed(text)

    text
    text.shape
    text.argmax(dim=-1)
    
    x.shape[0]
    torch.arange(x.shape[0]), text.argmax(dim=-1)
    x.shape
    x[0, 57, :]
    x[0, 57]
    x[torch.arange(x.shape[0]), text.argmax(dim=-1)][0, 0]

    image = torch.randn((batch_size, 3, 1080, 1920))
    # image_features = torch.randn((100, 512))
    # text_features = torch.randn((100, 768))

    # image_features = F.normalize(input=image_features, p=2, dim=1)
    # text_features = F.normalize(input=text_features, p=2, dim=1)

    # image_features
    # image_features @ text_features.t()