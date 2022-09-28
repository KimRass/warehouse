# "AttributeError: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular import)"
```sh
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip uninstall opencv-contrib-python-headless

pip3 install opencv-contrib-python==4.5.5.62
```

# Read Image
```python
# OpenCV
cv2.imread()

# Python Image Library
img = Image.open(fp)
```

# Get Image Size
```python
# OpenCV
height, width = img.shape

# Python Image Library
width, height = img.size
```

# Show Image
```python
cv2.imshow(winname, mat)
plt.imshow()
draw.show()
```

# Save Image
```python
# `bbox_inches="tight"`
plt.savefig()
fig.savefig()
```

# Resize Image
```python
cv2.resize(src, dsize, interpolation)
```

# Convert Color Map
```python
# OpenCV
# `code`: (`cv2.COLOR_BGR2GRAY`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2HSV`)
cv2.cvtColor(image, code)

# Python Image Library
# (`"RGB"`, `"RGBA"`, `"CMYK"`, `"L"`, `"1"`)
    # `"1"`, `"L"`: Binary
img.convert("L")
```

# Draw
```python
draw = ImageDraw.Draw(img)
```
## Draw Line
```python
...
draw.line([fill], [width])
```
## Draw Rectangle
- ![rectangle](https://img-blog.csdnimg.cn/c9e5333918814205a7cff8d65648784d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5Y2X6aOO5LiN56ueOg==,size_1,color_FFFFFF,t_70,g_se,x_15)
```python
# cv2
cv2.rectangle(img, pt1, pt2, color, thickness)

# Python Image Library
draw.rectangle(xy, outline, width)
```
## Draw Circle
```python
cv2.circle(img, center, radius, color, [thickness], [lineType], [shift])
```
## Draw Polygon
```python
# OpenCV
# Example
cv2.polylines(
    img=img,
    pts=[np.array(row[: 8]).reshape(-1, 2)],
    isClosed=True,
    color=(255, 0, 0),
    thickness=1
)
```
## Put Text
```python
# cv2
# Reference: https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
# `text`: Text string to be drawn.
# `org`: Bottom-left corner of the text string in the image.
# `fonFace`: (`cv2.FONT_HERSHEY_SIMPLEX`, ...) Font type.
# `fontScale`: Font scale factor that is multiplied by the font-specific base size.
# `thickness`: Font의 두께
cv2.putText(img, text, org, fontFace, fontScale, color, [thickness], [lineType])

# Python Image Library
draw.text(xy, text, fill, [font])
# Example
draw.text(
    xy=(bbox[0], bbox[1]),
    text=text_direction,
    fill=(100, 100, 100),
    font=ImageFont.truetype("Chalkduster.ttf", size=32)
)
```

# `cv2.getTextSize(text, fontFace, fontScale, thickness)`
```python
(text_width, text_height), baseline = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=font_scale, thickness=bbox_thick)
```

# Threshold Image
```python
# `maxval`: `thresh`를 넘었을 때 적용할 value.
# `type`:
    # `type=0`: `type=cv2.THRESH_BINARY`
# Returns the threshold that was used and the thresholded image.
_, img = cv.threshold(src, thresh, maxval, type)
```

# Connect?
- Reference: https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
```python
# `image`: Binary image.
# `n_label`: The total number of unique labels (i.e., number of total components) that were detected.
# `labels`: A mask named labels has the same spatial dimensions as our input binary image. For each location in labels, we have an integer ID value that corresponds to the connected component where the pixel belongs. You’ll learn how to filter the labels matrix later in this section.
# `stats`: Statistics on each connected component, including the bounding box coordinates and area (in pixels).
    # Shape: (`n_label`, 5)
    # 차례대로 `cv2.CC_STAT_LEFT`, `cv2.CC_STAT_TOP`, `cv2.CC_STAT_WIDTH`, `cv2.CC_STAT_HEIGHT`, `cv2.CC-STAT_AREA`
# `centroids`: The centroids (i.e., center) (x, y)-coordinates of each connected component.
    # Shape: (`n_label, 2)
n_label, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity)
```

# Morphology
```python
# `ksize`: kernel의 크기
cv2.getStructureingElement(shape, ksize, [anchor])
# Example
kernel = cv2.getStructureingElement(shape=cv2.MORPH_RECT, ksize, [anchor])
# `src`: Binary image

# Thinning: 하얀색 영역이 줄어듭니다.
cv2.erode(src, kernel)
# Thickening: 하얀색 영역이 증가합니다.
cv2.dilate(src, kernel)
```

# Apply Colormap
```python
# (height, width) -> (height, width, 3)
cv2.applyColorMap(src, colormap)
```

# Get Bounding Rectangle
```python
cv2.boundingRect()
```

# Get Bounding Minimum Area Rectangle
- Reference: https://theailearner.com/tag/cv2-minarearect/
```python
# The bounding rectangle is drawn with a minimum area. Roation is considered.
# Retuns a Box2D struecture which contains "(Center(x, y), (Width, Height), Angle of rotation)".
rectangle = cv2.minAreaRect(np_contours)
# Converts to four corners of the rectangle.
# Four corner points are ordered clockwise starting from the point with the highest y. If two points have the same highest y, then the rightmost point is the starting point.
# Angle of rotation is the angle between line joining the starting and endpoint and the horizontal.
box = cv2.boxPoints(rectangle)
```

# Zoom Array
```python
# The array is zoomed using spline interpolation of the requested order.
# `zoom`: The zoom factor along the axes. If a float, `zoom` is the same for each axis. If a sequence, `zoom` should contain one value for each axis.
# `order`: The order of the spline interpolation, default is `3`. The order has to be in the range 0-5.
scipy.ndimage.zoom(input, zoom, [order=3])
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