# Convert all stacks to meshes


import os
import sys


from stack_to_mesh import stack_to_mesh


if __name__ == "__main__":
    args = sys.argv[1:]
    input_folder = str(args[0])
    output_folder = args[1]

    sub_dirs = [x[0] for x in os.walk(input_folder)]
    sub_dirs = sub_dirs[1:]

    for sub_dir in sub_dirs[2:]:
        print(sub_dir)

        stack_to_mesh(sub_dir, output_folder)