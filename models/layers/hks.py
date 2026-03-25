import numpy as np
from pygel3d import hmesh
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import *

def cotan_weight(p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray) -> float:
    """cotan of the angle to k in triangle (i,j,k)"""
    u = p_i - p_k
    v = p_j - p_k
    cot = np.dot(u, v) / np.linalg.norm(np.cross(u, v))
    return cot

def laplace_beltrami_matrix_f(mesh: hmesh.HMesh) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Computes the cotangent Laplacian and M is the mass matrix (area-based)
        mesh: input mesh
    Returns:
        L: cotangent Laplacian matrix
        M: mass matrix (area-based)
    """
    vertices = mesh.vertices()
    pos = mesh.positions()
    n_v = len(vertices)
    L = np.zeros(shape=(n_v, n_v))
    M_diag = np.zeros(n_v)
    for i in range(n_v):
        v_i = i
        
        faces_adj = mesh.circulate_vertex(v_i, mode="f")
        area_sum = 0
        for f in faces_adj:
            area_sum += mesh.area(f)
        M_diag[i] = area_sum / 3.0
        
        neighbors = mesh.circulate_vertex(v_i, mode="v")
        sum_weights = 0
        
        for v_j in neighbors:
            faces_ij = list(set(mesh.circulate_vertex(v_i, mode="f")) & 
                            set(mesh.circulate_vertex(v_j, mode="f")))
        
            w_ij = 0
            
            for face in faces_ij:
                f_verts = list(mesh.circulate_face(face, mode="v"))
                v_k = [v for v in f_verts if v != v_i and v != v_j][0]
                
                w_ij += cotan_weight(pos[v_i], pos[v_j], pos[v_k])
                
            w_ij = 0.5 * w_ij
            L[v_i, v_j] = -w_ij
            sum_weights += w_ij
        
        L[v_i, v_i] = sum_weights
    
    return L, np.diag(M_diag)

def eigen_decomposition(L: np.ndarray, M: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the k smallest non-zero eigenvalues and corresponding eigenvectors of the generalized eigenvalue problem L * phi = lambda * M * phi.
        L: cotangent Laplacian matrix
        M: mass matrix (area-based)
        k: number of eigenvalues/eigenvectors to compute
    Returns:
        eval: array of eigenvalues
        evecs: matrix of eigenvectors (each column is an eigenvector)
    """
    # sparse matrices for efficient eigenvalue computation
    L_sparse = sparse.csc_matrix(L)
    M_sparse = sparse.csc_matrix(M)

    # We solve L * phi = lambda * M * phi
    # 'SM' is for 'Smallest Magnitude' eigenvalues, but since we want the smallest non-zero eigenvalues, we use 'sigma=0' to shift the spectrum
    evals, evecs = eigsh(L_sparse, k=k, M=M_sparse, sigma=0, which='LM')
    
    return evals, evecs

def compute_hks(evals: np.ndarray, evecs: np.ndarray, t: float = 0.01) -> np.ndarray:
    """
    Computes the Heat Kernel Signature (HKS) for each vertex given the eigenvalues and eigenvectors of the Laplacian.
        evals: array of eigenvalues
        evecs: matrix of eigenvectors (each column is an eigenvector)
        t: time parameter for HKS
    Returns:
        hks: array of HKS values for each vertex
    """
    n_vertices = evecs.shape[0]
    k = len(evals)
    
    hks_result = np.zeros((n_vertices, k))
    for j in range(k):
        hks_result[:, j] = np.exp(-evals[j] * t) * evecs[:, j]**2
    
    hks = np.sum(hks_result, axis=1)
    
    return hks