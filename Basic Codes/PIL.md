# Image
```python
from PIL import Image
```
## Image.open()
```python
img = Image.open("20180312000053_0640 (2).jpg")
```
### img.size
### img.save()
### img.thumbnail()
```python
img.thumbnail((64, 64))  
```
### img.crop()
```python
img_crop = img.crop((100, 100, 150, 150))
```
### img.resize()
```python
img = img.resize((600, 600))
```
### img.convert()
```python
img.convert("L")
```
- "RGB" | "RGBA" | "CMYK" | "L" | "1"
### img.paste()
```python
img1.paste(img2, (20,20,220,220))
```
- img2.size와 동일하게 두 번째 parameter 설정.
## Image.new()
```python
mask = Image.new("RGB", icon.size, (255, 255, 255))
```
