```python
from PIL import Image, ImageDraw, ImageFont
```

# Read Image
```python
img = Image.open(fp)
```

# Save Image
```python
Image.save()
```

# Draw
```python
draw = ImageDraw.Draw(img)
```
## Draw Line
```python
draw.line([fill], [width])
```
## Draw Circle
```python
# Example
draw.ellipse((x-r, y-r, x+r, y+r), fill=(255, 0, 0))
```
## Draw Rectangle
```python
# `outline`: Color to use for the outline.
# `fill`: Color to use for the fill.
# `width`: The line width, in pixels.
# Example
draw.rectangle(xy=(xmin, ymin, xmax, ymax), outline="red", width=2)
```
## Draw Text
```python
# Example
draw.text(
    xy=(xmin, ymin - 4),
    text=str(idx).zfill(3),
    fill="white",
    stroke_fill="black",
    stroke_width=2,
    font=ImageFont.truetype(font="fonts/NanumSquareNeo-bRg.ttf", size=26),
    anchor="ls"
)
```