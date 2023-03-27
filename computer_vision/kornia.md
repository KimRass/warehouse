```python
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import RandomCrop, Resize
```

# Spatially Jitter Color Channels
```python
class SpatiallyJitterColorChannels(nn.Module):
    def __init__(self, shift=0):
        super().__init__()

        self.shift = shift

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        for batch in range(b):
            for ch in range(c):
                x[batch: batch + 1, ch: ch + 1, ...] = torch.roll(
                    x[batch: batch + 1, ch: ch + 1, ...],
                    shifts=(random.randint(-self.shift, self.shift), random.randint(-self.shift, self.shift)),
                    dims=(2, 3)
                )
        return x
```

# Resize
```python
# Example
Resize(size=256)
```

# Random Crop
```python
# Example
RandomCrop(size=(256, 224))
```
