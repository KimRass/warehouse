import json
import numpy as np
import cv2
import skimage


with open("/Users/jongbeom.kim/Downloads/3D Vision Engineer 샘플데이터/annotation.json", mode="r") as f:
    loaded = json.load(f)


img = load_image_as_array("/Users/jongbeom.kim/Downloads/3D Vision Engineer 샘플데이터/image.png")
canvas = get_canvas_same_size_as_image(img=img, black=True)
for obj in loaded["shapes"]:
    label = obj["label"]
    points = obj["points"]

    cv2.fillPoly(
        img=canvas,
        pts=[np.array(points).astype("int")],
        color=(255, 255, 255)
    )
masked_img = get_masked_image(img=img, mask=canvas)


# img = load_image_as_array("/Users/jongbeom.kim/Downloads/3D Vision Engineer 샘플데이터/image.png")
depth = np.load("/Users/jongbeom.kim/Downloads/3D Vision Engineer 샘플데이터/depth.npy")
depth = cv2.cvtColor(depth, cv2.CV_16U)
cv2.imwrite("/Users/jongbeom.kim/Downloads/3D Vision Engineer 샘플데이터/depth.png", depth)

# cv2.imread("/Users/jongbeom.kim/Downloads/3D Vision Engineer 샘플데이터/depth.png", cv2.IMREAD_UNCHANGED)


stretch = skimage.exposure.rescale_intensity(depth, in_range="image", out_range=(0, 255)).astype(np.uint8)

temp = cv2.applyColorMap(src=depth, colormap=cv2.COLORMAP_OCEAN)
show_image(temp)



import matplotlib.pyplot as plt
import pyvista as pv
from pyvista import examples

# Load an interesting example of geometry
mesh = examples.load_random_hills()
type(mesh)

# Establish geometry within a pv.Plotter()
p = pv.Plotter()
p.add_mesh(mesh, color=True)
p.store_image = True  # permit image caching after plotter is closed
p.show()




stretch = skimage.exposure.rescale_intensity(depth, in_range='image', out_range=(0,255)).astype(np.uint8)
stretch.max()
show_image(stretch)

# convert to 3 channels
stretch = cv2.merge([stretch,stretch,stretch])

# define colors
color1 = (0, 0, 255)     #red
color2 = (0, 165, 255)   #orange
color3 = (0, 255, 255)   #yellow
color4 = (255, 255, 0)   #cyan
color5 = (255, 0, 0)     #blue
color6 = (128, 64, 64)   #violet
colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
colorArr.shape

# resize lut to 256 (or more) values
lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)

# apply lut
result = cv2.LUT(stretch, lut)
show_image(stretch)
lut.shape
stretch.shape

# create gradient image
grad = np.linspace(0, 255, 512, dtype=np.uint8)
grad = np.tile(grad, (20,1))
grad = cv2.merge([grad,grad,grad])

# apply lut to gradient for viewing
grad_colored = cv2.LUT(grad, lut)

# save result
cv2.imwrite('dist_img_colorized.png', result)
cv2.imwrite('dist_img_lut.png', grad_colored)

# display result
cv2.imshow('RESULT', result)
cv2.imshow('LUT', grad_colored)
cv2.waitKey(0)
cv2.destroyAllWindows()