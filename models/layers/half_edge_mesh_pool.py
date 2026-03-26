# Performs pooling operations.

"""
    Sorts the half-edges beased on the norms of their features and performs pooling operations
    until the mesh is small enough (number of half-edges is smaller than or equal to the target).
    The pooling operation is performed on the neighborhood of the half-edge and its opposite half-edge. (WHY?)
    The neighborhood is defined as the one-ring of the vertices of the half-edge and its opposite half-edge.
    The pooling operation is performed by the HalfEdgeNeighborhood class. After pooling, cleanup operations 
    are performed to ensure that the mesh remains topologically valid. Finally, the features of the remaining 
    half-edges are averaged to obtain the output features.
"""

from heapq import heappop, heapify
import numpy as np
import torch
import os
import torch.nn as nn

from models.layers import mesh_union
from models.layers.half_edge_neighborhood import HalfEdgeNeighborhood

class HalfEdgeMeshPool(nn.Module):

    def __init__(self, half_edge_target):
        super(HalfEdgeMeshPool, self).__init__()
        self.__half_edge_target = half_edge_target
        
        # directory where pre/post pool npz will be written (env var or default)
        self._export_pooled_dir = os.environ.get('EXPORT_POOLED_DIR', None)
        if self._export_pooled_dir:
            os.makedirs(self._export_pooled_dir, exist_ok=True)

    def __call__(self, half_edge_features, meshes):
        return self.forward(half_edge_features, meshes)

    def forward(self, half_edge_features, meshes):
        """ This method performs the pooling for all meshes in the batch. It creates the queue 
            and then performs the pooling operations until the desired target is reached. It then 
            performs cleanup operations and averages features.
        """
        updated_half_edge_features = []
        
        #only in test, not train
        if self._export_pooled_dir:
            # chosen channel (env var) or default 0
            ch_env = os.environ.get('EXPORT_POOLED_CHANNEL', None)
            try:
                export_channel = None if ch_env is None else (int(ch_env) if ch_env != 'all' else 'all')
            except Exception:
                export_channel = 0

            # number of pre-pool snapshots to save per mesh (4 pools -> 4 pre-pool files)
            NUM_EXPORT_PRE_STAGES = 4
            
        for mesh_index, mesh in enumerate(meshes):
            # ensure per-mesh stage counter exists (0..n_pool_layers-1)
            stage = getattr(mesh, '_export_pool_stage', 0)

            # features for this mesh (tensor C x N)
            half_edge_features_of_mesh_without_padding = half_edge_features[mesh_index, :, :mesh.half_edge_count]

            if self._export_pooled_dir:
                # --- save ONLY pre-pool selected channel as 1D array (one value per half-edge)
                # Save at most NUM_EXPORT_PRE_STAGES per mesh (stage numbers 0..NUM_EXPORT_PRE_STAGES-1)
                try:
                    if self._export_pooled_dir and stage < NUM_EXPORT_PRE_STAGES:
                        arr = half_edge_features_of_mesh_without_padding.detach().cpu().numpy()
                        arr = np.asarray(arr, dtype=np.float32)
                        # ensure canonical (C, N)
                        if arr.ndim == 2:
                            C, N = arr.shape
                            # if shape looks like (N, C) instead, transpose
                            if C == mesh.half_edge_count and N != mesh.half_edge_count:
                                arr = arr.T
                                C, N = arr.shape
                        else:
                            # flatten trailing dims into channels x N
                            arr = arr.reshape(arr.shape[0], -1)
                            C, N = arr.shape

                        # select channel or reduce across channels
                        if export_channel == 'all':
                            sel = np.linalg.norm(arr, axis=0).astype(np.float32)
                            ch_label = 'all'
                        else:
                            ch = 0 if export_channel is None else int(export_channel)
                            if ch < arr.shape[0]:
                                sel = arr[ch, :mesh.half_edge_count].astype(np.float32)
                            else:
                                sel = np.linalg.norm(arr, axis=0).astype(np.float32)
                            ch_label = str(ch)

                        # sanitize non-finite values and ensure length equals half_edge_count
                        sel = np.nan_to_num(sel, nan=0.0, posinf=1e6, neginf=-1e6)
                        if sel.size != mesh.half_edge_count:
                            sel = sel[:mesh.half_edge_count] if sel.size > mesh.half_edge_count else np.pad(sel, (0, mesh.half_edge_count - sel.size), mode='constant', constant_values=0.0)

                        if stage == 0:
                            pre_fname = os.path.join(self._export_pooled_dir, f"{mesh._HalfEdgeMesh__filename.split('.')[0]}_initial_hks.npy")
                        else:
                            pre_fname = os.path.join(self._export_pooled_dir, f"{mesh._HalfEdgeMesh__filename.split('.')[0]}_s{stage}_ch{ch_label}.npy")
                        np.save(pre_fname, sel)
                except Exception:
                    pass

            heap = self.__build_queue(half_edge_features_of_mesh_without_padding, mesh.half_edge_opposite)

            half_edge_mask = np.ones(mesh.half_edge_count, dtype=np.bool)
            half_edge_groups = mesh_union.MeshUnion(mesh.half_edge_count, half_edge_features.device)

            while mesh.half_edge_count > self.__half_edge_target:
                _, half_edge_id, opposite_half_edge_id = heappop(heap)
                self.__pool_neighborhoods(mesh, half_edge_id, opposite_half_edge_id, half_edge_mask, half_edge_groups)

            mesh.correct_data_after_pooling(half_edge_mask, half_edge_groups)

            updated_half_edge_features_of_mesh = half_edge_groups.rebuild_features(half_edge_features[mesh_index], half_edge_mask, self.__half_edge_target)

            # increment the mesh export stage so subsequent pool layers write stage+1 files
            mesh._export_pool_stage = stage + 1

            updated_half_edge_features.append(updated_half_edge_features_of_mesh)
 
        # create tensor from list of tensors
        out_features = torch.cat(updated_half_edge_features)
        out_features = out_features.view(len(meshes), -1, self.__half_edge_target)
        return out_features


    # Creates a heap of (half-edge, opposite) of the mesh with their corresponding norms as keys. The heap is used to determine the order of pooling.
    def __build_queue(self, half_edge_features, opposites):
        squared_magnitude = torch.sum(half_edge_features*half_edge_features, 0).squeeze().tolist()

        half_edge_pairs = [tuple(sorted([i, opposites[i]])) for i in range(len(opposites))]
        half_edge_pairs = set(half_edge_pairs)

        norms_with_he_pairs = []
        for index_edge_0, index_edge_1  in half_edge_pairs:
            sum_norm = squared_magnitude[index_edge_0] + squared_magnitude[index_edge_1]
            combined_norm_with_he_pair = [sum_norm / 2, index_edge_0, index_edge_1]

            norms_with_he_pairs.append(combined_norm_with_he_pair)

        heapify(norms_with_he_pairs)

        return norms_with_he_pairs


    def __pool_neighborhoods(self, mesh, half_edge_id, opposite_id, half_edge_mask, half_edge_groups):
        """ Cleans the faces of the half-edge and the opposite half_edge, checks if they are suitable for pooling
            and performs the pooling of the half-edge pooling neighborhood if successfull.
            Interrupts procedure if half-edge-target was reached at suitable moments."""
        if          half_edge_mask[half_edge_id] \
                and half_edge_mask[opposite_id]\
                and self.__clean_face_of_halfedge(mesh, half_edge_id, half_edge_mask, half_edge_groups) \
                and self.__clean_face_of_halfedge(mesh, opposite_id,  half_edge_mask, half_edge_groups) \
                and self.__check_connectivity_of_one_ring_vertices(mesh, half_edge_id):
            half_edge_neighborhood_he = HalfEdgeNeighborhood(mesh, half_edge_id)
            half_edge_neighborhood_he.pool(half_edge_mask, half_edge_groups)

            half_edge_neighborhood_opp = HalfEdgeNeighborhood(mesh, opposite_id)
            half_edge_neighborhood_opp.pool(half_edge_mask, half_edge_groups)

            # We need to remove the half_edge_id and opposite_id here, because the data of opposite is used in pooling.
            mesh.remove_half_edge(half_edge_id)
            mesh.remove_half_edge(opposite_id)

            mesh.merge_vertices_of_halfedge(half_edge_id)

            half_edge_mask[half_edge_id] = False
            half_edge_mask[opposite_id] = False


    def __clean_face_of_halfedge(self, mesh, half_edge_id, half_edge_mask, half_edge_groups):
        """ Collapsing edges that violate the link condition (Dey et al. 1998)
            can cause a mesh to become topologically irregular. To avoid this, this method checks before
            edge collapses whether the to-be-collapsed edge violates the link condition and restores it if violated
            using a 3:1 triangle collapse. Since the resulting face might also violate the link condition, this process
            is repeated until the face of the half-edge does not violate the link-condition or the pooling target is reached.
            Returns False if __half_edge_target was reached during cleaning, True otherwise.
        """
        while True:
            # For the check we need the indices of the half-edges of the adjacent faces of the next and the previous halfedge.
            half_edge_neighborhood = HalfEdgeNeighborhood(mesh, half_edge_id)
            if half_edge_neighborhood.violates_link_condition() and mesh.half_edge_count > self.__half_edge_target:
                half_edge_neighborhood.perform_1_3_triangle_collapse(half_edge_groups, half_edge_mask)
            else:
                break

        if mesh.half_edge_count <= self.__half_edge_target:
            return False
        return True


    @staticmethod
    def __check_connectivity_of_one_ring_vertices(mesh, half_edge_id):
        """ Checks whether the vertices of the half-edge have two shared vertices in their one rings.
            If this is not the case, the connectivity is not suitable for pooling.
        """
        vertex_1_of_he = mesh.half_edges[half_edge_id, 0]
        vertex_2_of_he = mesh.half_edges[half_edge_id, 1]

        half_edges_of_vertex_1 = mesh.vertex_to_half_edges[vertex_1_of_he]
        half_edges_of_vertex_2 = mesh.vertex_to_half_edges[vertex_2_of_he]

        vertices_of_one_ring_of_vertex_1_incl_vertex1 = set(mesh.half_edges[half_edges_of_vertex_1].reshape(-1))
        vertices_of_one_ring_of_vertex_2_incl_vertex2 = set(mesh.half_edges[half_edges_of_vertex_2].reshape(-1))

        shared_vertices_with_he_vertices = vertices_of_one_ring_of_vertex_1_incl_vertex1 & vertices_of_one_ring_of_vertex_2_incl_vertex2

        shared_vertices_without_he_vertices = shared_vertices_with_he_vertices - set(mesh.half_edges[half_edge_id])

        if len(shared_vertices_without_he_vertices) == 2:
            return True
        else:
            return False
