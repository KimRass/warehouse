# Watershed Algorithm
- Reference: https://webnautes.tistory.com/1281
- ![watershed](https://imagej.net/media/plugins/watershed-flooding-graph.png)
```python
def apply_watershed(img, mask):
    kernel = np.ones((2, 6), np.uint8)
    sure_bg = cv2.dilate(src=mask[:, :, 0], kernel=kernel, iterations=7)

    dist_transform = cv2.distanceTransform(src=mask[:, :, 0], distanceType=cv2.DIST_L2, maskSize=5)
    _, sure_fg = cv2.threshold(
        src=dist_transform, thresh=0.07 * dist_transform.max(), maxval=255, type=cv2.THRESH_BINARY
    )

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers += 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    img_copied = copy.deepcopy(img)
    markers = cv2.watershed(img_copied, markers)

    img_copied[markers == -1] = [255, 0, 0]
    img_copied[markers == 1] = [255, 255, 0]
    return markers, img_copied
```