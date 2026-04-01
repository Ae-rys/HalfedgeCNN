import polyscope
import numpy as np
from pygel3d import hmesh

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser("view meshes")
    parser.add_argument('--file', required=True, type=str,
                        help="list of 1 or more .obj files")
    parser.add_argument('--hks_values', nargs='*', default=None, type=str,
                        help="list of .npy/.npz files with half-edge values (one per mesh)")
    args = parser.parse_args()
    
    mesh = hmesh.load("checkpoints/shrec_16/export/classification/" + args.file)
    
    allfaces = [[v for v in mesh.circulate_face(f)] for f in mesh.faces()]

    polyscope.init()
    polyscope.register_surface_mesh("My Mesh", vertices=np.array(mesh.positions()), faces=allfaces)
    
    if args.hks_values is not None:
        hks = np.load("checkpoints/shrec_16/export/classification/" + args.hks_values[0])
        print("shape of hks:", hks.shape)
        polyscope.get_surface_mesh("My Mesh").add_scalar_quantity("Halfedge_values", hks.flatten(), defined_on="halfedges")
    
    polyscope.show()
