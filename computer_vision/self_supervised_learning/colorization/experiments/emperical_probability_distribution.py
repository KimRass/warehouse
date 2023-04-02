import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

data_dir = "/Users/jongbeomkim/Downloads/imagenet-mini/train"
data_dir = Path(data_dir)
stacked_a = np.empty((0,), dtype="float64")
stacked_b = np.empty((0,), dtype="float64")
for img_path in data_dir.glob("**/*.JPEG"):
    img = load_image(img_path)
    img_lab = rgb2lab(img)
    a = img_lab[..., 1].flatten()
    b = img_lab[..., 2].flatten()

    stacked_a = np.concatenate([stacked_a, a], axis=0)
    stacked_b = np.concatenate([stacked_b, b], axis=0)
smaller = min(stacked_a.shape[0], stacked_b.shape[0])
stacked_a = stacked_a[: smaller]
stacked_b = stacked_b[: smaller]
stacked_a.shape

heatmap, xedges, yedges = np.histogram2d(stacked_a, stacked_b, bin_idx_map=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.colorbar(ScalarMappable(norm=None, cmap='viridis'))
plt.xscale('log')
plt.yscale('log')
plt.show()