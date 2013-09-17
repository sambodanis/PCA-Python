#!/usr/bin/env python
import numpy
from scipy import linalg


class PCA:
    
    def __init__(self, mat, variance_to_keep):
        mat /= -1.0 * numpy.std(mat)
        mat -= numpy.mean(mat)
        covariance = mat.T.dot(mat)
        mat_r, mat_c = mat.shape
        covariance = covariance * (1.0 / mat_r)
        U, s, V = linalg.svd(covariance)
        k = 1
        variation = 1.0
        trace_total = sum(s)
        while (variation >= 1 - variance_to_keep / 100.0):
            trace_k = 0.0
            variation = 1.0 - sum(s[:k]) / trace_total
            k += 1
        self.principal_components = U[:, :k]
        Z = self.principal_components.T.dot(mat.T)
        self.compressed_matrix = self.principal_components.dot(Z).T

    def get_principal_components(self):
        return self.principal_components
    
    def get_compressed_matrix(self):
        return self.compressed_matrix
    