img_in_bbox = img[y_min: y_max, x_min: x_max]

w = x_max - x_min + 1
h = y_max - y_min + 1

# Width와 Height 중 더 작은 것이 20이 됩니다.
if w < h:
    rate = 20.0 / w
    w = int(round(w * rate))
    h = int(round(h * rate / 20.0) * 20)
else:
    rate = 20.0 / h
    w = int(round(w * rate / 20.0) * 20)
    h = int(round(h * rate))
img_in_bbox = cv2.resize(img_in_bbox, dsize=(w, h))

mat = np.zeros(shape=(1, h, w), dtype="uint8")
mat[0, :, :] = 0.299 * img_in_bbox[:, :, 0] + 0.587 * img_in_bbox[:, :, 1] + 0.114 * img_in_bbox[:, :, 2]

xx_pad = mat.astype(np.float32) / 255.
xx_pad = torch.from_numpy(
    np.expand_dims(xx_pad, axis=0)
) # (1, height, width) -> (1, 1, height, width)
if cuda:
    xx_pad.cuda()