import kagglehub

# Download latest version
path = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset", output_dir="datasets/kaggle_modelnet40")

print("Path to dataset files:", path)
