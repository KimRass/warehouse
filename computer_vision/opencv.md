# Read Image
```python
def load_image_as_array(url_or_path="", gray=False):
    url_or_path = str(url_or_path)

    if "http" in url_or_path:
        img_arr = np.asarray(
            bytearray(requests.get(url_or_path).content), dtype="uint8"
        )
        if not gray:
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)
    else:
        if not gray:
            img = cv2.imread(url_or_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(url_or_path, cv2.IMREAD_GRAYSCALE)
    return img
```

# Save Image
```python
def convert_to_array(img):
    img = np.array(img)
    return img


def blend_two_images(img1, img2, alpha=0.5):
    img1 = convert_to_pil(img1)
    img2 = convert_to_pil(img2)
    img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return img_blended


def save_image(img1, path, img2=None, alpha=0.5) -> None:
    if img2 is None:
        img_arr = convert_to_array(img1)
    else:
        img_arr = convert_to_array(
            blend_two_images(img1=img1, img2=img2, alpha=alpha)
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if img_arr.ndim == 3:
        cv2.imwrite(
            filename=str(path), img=img_arr[:, :, :: -1], params=[cv2.IMWRITE_JPEG_QUALITY, 100]
        )
    elif img_arr.ndim == 2:
        cv2.imwrite(
            filename=str(path), img=img_arr, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
        )
```

# Resize Image
```python
# `dsize`: (width, height)
cv2.resize(src, dsize, interpolation)
```

# Threshold Image
```python
# `maxval`: `thresh`를 넘었을 때 적용할 value.
# `type`:
    # `type=0`: `type=cv2.THRESH_BINARY`
# Returns the threshold that was used and the thresholded image.
_, img = cv2.threshold(src, thresh, maxval, type)
```

# Morphology
```python
# `shape`: `cv2.MORPH_ELLIPSE`, `cv2.MORPH_RECT`, ...
# `ksize`: kernel의 크기
cv2.getStructuringElement(shape, ksize, [anchor])

# `src`: Binary image
# Thinning: 하얀색 영역이 줄어듭니다.
cv2.erode(src, kernel)
# Thickening: 하얀색 영역이 증가합니다.
cv2.dilate(src, kernel)
```
- Opening: erosion followed by dilation
- Closing: Dilation followed by Erosion

# Colormap
## Convert Color Map
```python
# `code`: (`cv2.COLOR_BGR2GRAY`, `cv2.COLOR_BGR2RGB`, `cv2.COLOR_BGR2HSV`)
cv2.cvtColor(image, code)
```
## Apply Colormap
```python
# (height, width) -> (height, width, 3)
cv2.applyColorMap(src, colormap)
```

# Draw
## Draw Polygon
```python
# Example
cv2.polylines(
    img=img,
    # `"int64"`
    pts=[np.array(row[: 8]).reshape(-1, 2)],
    isClosed=True,
    color=(255, 0, 0),
    thickness=1
)
```
## Draw Filled Polygon
```python
# Example
cv2.fillPoly(
    img=canvas,
    pts=[np.array(points).astype("int64")],
    color=(255, 255, 255)
)
```
## Draw Rectangle
```python
cv2.rectangle(img, pt1, pt2, color, thickness)
```
## Draw Circle
```python
# `thickness=-1`: Fill circle
cv2.circle(img, center, radius, color, [thickness], [lineType], [shift])
```
## Draw Text
```python
# Reference: https://docs.opencv2.org/4.x/d6/d6e/group__imgproc__draw.html#ga0f9314ea6e35f99bb23f29567fc16e11
# `text`: Text string to be drawn.
# `org`: Bottom-left corner of the text string in the image.
# `fonFace`: (`cv2.FONT_HERSHEY_SIMPLEX`, ...) Font type.
# `fontScale`: Font scale factor that is multiplied by the font-specific base size.
# `thickness`: Font의 두께
# `bottomLeftOrigin`:
    # `True`: 문자를 위아래로 뒤집습니다.
    # `False`: (Default)
cv2.putText(img, text, org, fontFace, fontScale, color, [thickness], [lineType], [bottomLeftOrigin])
```
### Get Text Size
```python
(text_width, text_height), baseline = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=font_scale, thickness=bbox_thick)
```

# Contour
## Find Contours
```python
contours, _ = cv2.findContours(image=overlap_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
```
## Draw Contours
```python
cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
```

# Connected Component Labeling
- Reference: https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
```python
# `image`: Binary image.
# `n_label`: The total number of unique labels (i.e., number of total components) that were detected.
# `labels`: A mask named labels has the same spatial dimensions as our input binary image. For each location in labels, we have an integer ID value that corresponds to the connected component where the pixel belongs. You’ll learn how to filter the labels matrix later in this section.
# `stats`: Statistics on each connected component, including the bounding box coordinates and area (in pixels).
    # Shape: (`n_label`, 5)
    # 차례대로 `cv2.CC_STAT_LEFT`, `cv2.CC_STAT_TOP`, `cv2.CC_STAT_WIDTH`, `cv2.CC_STAT_HEIGHT`, `cv2.CC_STAT_AREA`
# `centroids`: The centroids (i.e., center) (x, y)-coordinates of each connected component.
    # Shape: (`n_label, 2)
# Reference: https://stackoverflow.com/questions/7088678/4-connected-vs-8-connected-in-connected-component-labeling-what-is-are-the-meri
    # What will happen is that using 4 connected for labelling, you will probably get more objects. It's like an island of pixels. Some 'islands' are connected with others islands by only one pixel, and if this pixel is in diagonal, using 4 connected will label both islands as two separate objects, while 8 connected will assume they are only one object.
n_label, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity)
```

```python
_, text_mask = cv2.threshold(src=text_score_map, thresh=120, maxval=255, type=cv2.THRESH_BINARY)
_, text_segmap, stats, _ = cv2.connectedComponentsWithStats(image=text_mask, connectivity=4)

heights = stats[1:, cv2.CC_STAT_HEIGHT]

interval = 4
for value in range(heights.max() // interval + 1):
    labels = np.argwhere((heights >= interval * value) & (heights < interval * (value + 1)))

    temp = np.isin(text_segmap, labels.flatten()).astype("uint8") * 255
    show_image(temp)
    
    show_image(temp)
```

# Get Bounding Rectangle
```python
cv2.boundingRect()
```
```python
def get_minimum_area_bbox(mask):
    if mask.ndim == 3:
        mask = np.sum(mask, axis=2)

    nonzero_row, nonzero_col = np.nonzero(mask)
    nonzero_row = np.sort(nonzero_row)
    nonzero_col = np.sort(nonzero_col)
    
    ymin = nonzero_row[0]
    ymax = nonzero_row[-1]
    xmin = nonzero_col[0]
    xmax = nonzero_col[-1]
    return xmin, ymin, xmax, ymax
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

# Rotate Image
```python
# Example
patch = cv2.rotate(src=patch, rotateCode=cv2.ROTATE_90_CLOCKWISE)
```

# Get Mask in Color Boundary
```python
# Example
mask = cv2.inRange(src=mask, lowerb=(255, 100, 100), upperb=(255, 100, 100))
```
