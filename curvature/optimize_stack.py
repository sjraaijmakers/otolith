# Apply several filters to PD to optimize for H derivation

import sys
sys.path.insert(1, '../functions')

import vtk_functions
from vtk.util import numpy_support
import meshlabxml as mlx
import numpy as np
import os


def stack_to_mesh(input_folder, mesh_folder, sigma=3):
    basename = os.path.basename(input_folder)

    # VTK
    imgdata = vtk_functions.folder_to_imgdata(input_folder)

    if sigma > 0:
        padding = sigma * 4
        imgdata = vtk_functions.pad_imgdata(imgdata, padding)
        imgdata = vtk_functions.gauss_imgdata(imgdata, sigma=sigma)
        print("Applied Gaussian smoothing with s=%s" % sigma)
        basename = basename + "_g_%s" % sigma

    ply_file_name =  "%s/%s.ply" % (mesh_folder, basename)
    polydata = vtk_functions.imgdata_to_pd(imgdata)
    vtk_functions.write_ply(polydata, ply_file_name)

    # MESHLAB
    input_mesh = ply_file_name
    output_mesh = "%s/%s_MLX.ply" % (mesh_folder, basename)

    mesh_topology = mlx.files.measure_topology(input_mesh)
    target_faces = int(np.round(mesh_topology["face_num"] * target_reduction))

    simplified_mesh = mlx.FilterScript(file_in=input_mesh, file_out=output_mesh)

    # Apply decimation

    t_faces = int(np.round(mesh_topology["face_num"] * 0.5 ** 1))
    mlx.remesh.simplify(simplified_mesh,
                            texture=False,
                            faces=t_faces,
                            quality_thr=1,
                            preserve_topology=False,
                            preserve_boundary=True)

    t_faces = int(np.round(mesh_topology["face_num"] * 0.5 ** 2))
    mlx.remesh.simplify(simplified_mesh,
                            texture=False,
                            faces=t_faces,
                            quality_thr=1,
                            preserve_topology=False,
                            preserve_boundary=True)

    t_faces = int(np.round(mesh_topology["face_num"] * 0.5 ** 3))
    mlx.remesh.simplify(simplified_mesh,
                            texture=False,
                            faces=t_faces,
                            quality_thr=1,
                            preserve_topology=False,
                            preserve_boundary=True)

    t_faces = int(np.round(mesh_topology["face_num"] * 0.5 ** 4))
    mlx.remesh.simplify(simplified_mesh,
                            texture=False,
                            faces=t_faces,
                            quality_thr=1,
                            preserve_topology=False,
                            preserve_boundary=True)

    # Apply Taubin smoothing
    mlx.smooth.taubin(simplified_mesh, iterations=smoothing_iterations)

    simplified_mesh.run_script()


if __name__ == "__main__":
    # input_folder = "/home/steven/scriptie/inputs/edited_scans/otoM203_0.6"
    input_folder = "/home/steven/scriptie/inputs/edited_scans/otoI48"
    mesh_folder = "/home/steven/scriptie/inputs/meshes"

    stack_to_mesh(input_folder, mesh_folder, sigma=3)
