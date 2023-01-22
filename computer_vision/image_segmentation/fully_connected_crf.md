```python
import denseCRF
    Iq = img
    prob = np.dstack(
        [(segmap_text == label).astype("float32") for label in np.unique(segmap_text)]
    )

    w1    = 10.0  # weight of bilateral term
    alpha = 40    # spatial std
    beta  = 13    # rgb  std
    w2    = 3.0   # weight of spatial term
    gamma = 3     # spatial std
    it    = 10    # iteration
    lab = denseCRF.densecrf(Iq, prob, (w1, alpha, beta, w2, gamma, it))
    lab *= 255

    show_image(lab != 0)
```