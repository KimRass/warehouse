# Zoom Array
```python
# The array is zoomed using spline interpolation of the requested order.
# `zoom`: The zoom factor along the axes. If a float, `zoom` is the same for each axis. If a sequence, `zoom` should contain one value for each axis.
# `order`: The order of the spline interpolation, default is `3`. The order has to be in the range 0-5.
scipy.ndimage.zoom(input, zoom, [order=3])
```