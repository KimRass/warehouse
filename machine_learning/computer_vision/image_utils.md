# `cv2`
```bash
# On Windows
conda install -c conda-forge opencv
# On MacOS
pip install opencv-python
```
```python
import cv2
```
## `cv2.imread()`
## `plt.imshow()`
## `cv2.cvtColor(image, code)`
- `code`: (`cv2.COLOR_BGR2GRAY`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2HSV`)
## `cv2.resize(img, dsize, interpolation)`
## Draw Rectangle
```python
cv2.rectangle(img, pt1, pt2, color, thickness)
```
## `cv2.circle(img, center, radius, color, [thickness], [lineType], [shift])`
## `cv2.getTextSize(text, fontFace, fontScale, thickness)`
```python
(text_width, text_height), baseline = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=font_scale, thickness=bbox_thick)
```
## `cv2.putText(img, text, org, fontFace, fontScale, color, [thickness], [lineType])`
- Reference: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
- `text`: Text string to be drawn.
- `org`: Bottom-left corner of the text string in the image.
- `fonFace`: (`cv2.FONT_HERSHEY_SIMPLEX`, ...) Font type.
- `fontScale`: Font scale factor that is multiplied by the font-specific base size.

# `PIL`
```python
from PIL import Image, ImageDraw, ImageFont
```
## Open Image
```python
img = Image.open(fp)
```
### Manipulate Image
```python
img.size()
img.save()
img.thumbnail()
img.crop()
img.resize()

# (`"RGB"`, `"RGBA"`, `"CMYK"`, `"L"`, `"1"`)
img.convert("L")
```
### Paste Image
```python
# img2.size와 동일하게 두 번째 parameter 설정.	
img1.paste(img2, (20,20,220,220))
```
## `Image.new()`
```python
mask = Image.new("RGB", icon.size, (255, 255, 255))
```
## Draw Line
```python
draw = ImageDraw.Draw(img)
...
draw.line([fill], [width])
```

# Work with "HEIF" File
```sh
# Install
brew install libffi libheif
pip install git+https://github.com/carsales/pyheif.git
```
```python
# Landscape orientation refers to horizontal subjects or a canvas wider than it is tall. Portrait format refers to a vertical orientation or a canvas taller than it is wide.
```