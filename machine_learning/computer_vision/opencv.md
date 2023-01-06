# Draw Polygon
```python
# Example
cv2.polylines(
    img=img,
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
    pts=[np.array(points).astype("int")],
    color=(255, 255, 255)
)
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