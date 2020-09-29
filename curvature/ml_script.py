
import meshlabxml as mlx
import numpy as np


original_mesh = '/home/steven/scriptie/inputs/meshes/sphere_r_100_g_10_ij.ply' # input file
simplified_mesh = '/home/steven/scriptie/inputs/meshes/sphere_r_100_g_10_ij_MLX.ply' # output file

d = mlx.files.measure_topology(original_mesh)

target_faces = int(np.round(d["face_num"] * 0.5**4))

simplified_mesh = mlx.FilterScript(file_in=original_mesh, file_out=simplified_mesh) # Create FilterScript object
mlx.remesh.simplify(simplified_mesh, texture=False, faces=target_faces,
                 quality_thr=0.9, preserve_boundary=True)

mlx.smooth.taubin(simplified_mesh, iterations=40)

simplified_mesh.run_script() # Run the script
