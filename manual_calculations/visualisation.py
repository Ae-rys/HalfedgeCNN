# To visualise the data in kaggle_modelnet40

import trimesh

mesh = trimesh.load("datasets/kaggle_modelnet40/ModelNet40/airplane/test/airplane_0627.off")
mesh.show()


# To save in .obj
mesh.export('modele.obj')