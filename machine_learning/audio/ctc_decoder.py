import numpy as np
from ctc_decoder import best_path, beam_search

mat = np.array([[0.4, 0, 0.5, 0.1], [0.3, 0, 0.6, 0.1]])
mat.shape
chars = 'abc'

print(f'Best path: "{best_path(mat, chars)}"')
print(f'Beam search: "{beam_search(mat, chars)}"')
