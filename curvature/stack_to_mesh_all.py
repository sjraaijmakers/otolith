import os

from stack_to_mesh import stack_to_mesh


if __name__ == "__main__":
    input_folder = "/home/steven/scriptie/inputs/edited_scans"
    mesh_folder = "/home/steven/scriptie/inputs/meshes/oto"
    sub_dirs = [x[0] for x in os.walk(input_folder)]
    sub_dirs = sub_dirs[1:]

    for sub_dir in sub_dirs[0:2]:
        print(sub_dir)

        stack_to_mesh(sub_dir, mesh_folder)