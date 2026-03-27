# TODO: Implement time scales for HKS (multiple values of t for each halfedge)
""" I did not do it in the end, since with 1 value of t it was better for visualisation, and it already worked very well"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import *

def cotan_weight(p_i: np.ndarray, p_j: np.ndarray, p_k: np.ndarray) -> float:
    """cotan of the angle to k in triangle (i,j,k)"""
    u = p_i - p_k
    v = p_j - p_k
    cot = np.dot(u, v) / np.linalg.norm(np.cross(u, v))
    return cot

def get_faces_of_vertices(mesh_data):
    """ 
    Computes the faces adjacent to each vertex in the mesh.
        mesh_data: input mesh data containing the half-edge structure
    Returns:
        vertex_faces: list of lists, where vertex_faces[i] contains the indices of the faces adjacent to vertex i
    """
    vertex_faces = [[] for _ in range(len(mesh_data.vertex_positions))]
    
    for face_index, face in enumerate(mesh_data.faces):
        for vertex_index in face:
            vertex_faces[vertex_index].append(face_index)
    
    return vertex_faces

def get_vertices_of_faces(mesh_data):
    """ 
    Computes the vertices adjacent to each face in the mesh.
        mesh_data: input mesh data containing the half-edge structure
    Returns:
        face_vertices: list of lists, where face_vertices[i] contains the indices of the vertices adjacent to face i
    """
    face_vertices = [[] for _ in range(len(mesh_data.faces))]
    
    for face_index, face in enumerate(mesh_data.faces):
        for vertex_index in face:
            face_vertices[face_index].append(vertex_index)
    
    return face_vertices

def get_neighbors_of_vertices(mesh_data):
    """ 
    Computes the neighboring vertices for each vertex in the mesh.
        mesh_data: input mesh data containing the half-edge structure
    Returns:
        neighbors: list of lists, where neighbors[i] contains the indices of the neighboring vertices of vertex i
    """
    neighbors = [[] for _ in range(len(mesh_data.vertex_positions))]
    
    for half_edge in mesh_data.half_edges:
        v1 = half_edge[0]  # vertex at the start of the half-edge
        v2 = half_edge[1]  # vertex at the end of the half-edge
        if v2 not in neighbors[v1]:
            neighbors[v1].append(v2)
        if v1 not in neighbors[v2]:
            neighbors[v2].append(v1)
    
    return neighbors

def laplace_beltrami_matrix_f(mesh_data) -> Tuple[np.ndarray, np.ndarray]:
    """ 
    Computes the cotangent Laplacian and M is the mass matrix (area-based)
        mesh: input mesh
    Returns:
        L: cotangent Laplacian matrix
        M: mass matrix (area-based)
    """
    pos = mesh_data.vertex_positions
    n_v = len(pos)
    L = np.zeros(shape=(n_v, n_v))
    M_diag = np.zeros(n_v)
    faces_of_vertices = get_faces_of_vertices(mesh_data)
    vertices_of_faces = get_vertices_of_faces(mesh_data)
    neighbors_of_vertices = get_neighbors_of_vertices(mesh_data)
    for i in range(n_v):
        v_i = i
        
        faces_adj = faces_of_vertices[v_i]
        area_sum = 0
        for f in faces_adj:
            area_sum += mesh_data.face_areas[f]
        M_diag[i] = area_sum / 3.0
        
        neighbors = neighbors_of_vertices[v_i]
        sum_weights = 0
        
        for v_j in neighbors:
            faces_ij = list(set(faces_of_vertices[v_i]) & 
                            set(faces_of_vertices[v_j]))
        
            w_ij = 0
            
            for face in faces_ij:
                f_verts = vertices_of_faces[face]
                v_k = [v for v in f_verts if v != v_i and v != v_j][0]
                
                w_ij += cotan_weight(pos[v_i], pos[v_j], pos[v_k])
                
            w_ij = 0.5 * w_ij
            L[v_i, v_j] = -w_ij
            sum_weights += w_ij
        
        L[v_i, v_i] = sum_weights
    
    return L, np.diag(M_diag)

def eigen_decomposition(L: np.ndarray, M: np.ndarray, k: int = 100, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
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
    L_sparse = sparse.csc_matrix(L + eps * np.eye(L.shape[0]))  # I add small value to diagonal for numerical stability
    M_sparse = sparse.csc_matrix(M + eps * np.eye(M.shape[0])) 

    # We solve L * phi = lambda * M * phi
    # 'SM' is for 'Smallest Magnitude' eigenvalues, but since we want the smallest non-zero eigenvalues, we use 'sigma=0' to shift the spectrum
    evals, evecs = eigsh(L_sparse, k=k, M=M_sparse, sigma=0, which='LM')
    
    return evals, evecs

def compute_hks_vertices(evals: np.ndarray, evecs: np.ndarray, t: float = 0.01) -> np.ndarray:
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

def compute_hks_features(mesh_data, _):
    """
    Computes the HKS features for each vertex in the mesh.
        mesh_data: input mesh data containing the half-edge structure and vertex positions
        _: placeholder for compatibility with feature extractor interface
    Returns:
        hks_features: array of HKS features for each half-edge (shape: (1, num_half_edges))
    """
    L, M = laplace_beltrami_matrix_f(mesh_data)
    evals, evecs = eigen_decomposition(L, M, k=100)
    hks_features_vertices = compute_hks_vertices(evals, evecs, t=0.01)

    hks_features = np.zeros(len(mesh_data.half_edges))

    for i, half_edge in enumerate(mesh_data.half_edges):
        v1 = half_edge[0]
        v2 = half_edge[1]
        hks_features[i] = max(hks_features_vertices[v2], hks_features_vertices[v1]) # Seems to work better than the difference. i could try other things

    return np.expand_dims(hks_features, axis=0)