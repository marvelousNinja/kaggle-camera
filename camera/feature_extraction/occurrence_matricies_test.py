from camera.feature_extraction.occurrence_matricies import occurrence_matrix
from camera.shared.data import load_sample_image
import numpy as np

def test_occurrence_matrix():
    matrix = occurrence_matrix(load_sample_image())
    assert matrix.shape == (5, 5, 5)
    print(np.linalg.norm(matrix))
    print(matrix)
