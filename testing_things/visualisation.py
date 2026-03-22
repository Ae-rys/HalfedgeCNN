import trimesh

mesh = trimesh.load("datasets/kaggle_modelnet40/ModelNet40/airplane/test/airplane_0627.off")
mesh.show()


# Le sauvegarder en OBJ
mesh.export('modele.obj')