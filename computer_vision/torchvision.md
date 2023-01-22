# `make_gird()`
```python
temp = torchvision.utils.make_grid(img, nrow=10)
plt.imshow(temp.permute(1, 2, 0))
plt.axis("off")
plt.show()
```