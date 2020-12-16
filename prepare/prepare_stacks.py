# Bulk prepare stacks

import os
import sys
import prepare_stack


def has_subdirectories(dir):
    for f in os.scandir(s):
        if f.is_dir():
            return True

    return False


if __name__ == "__main__":
    args = sys.argv[1:]

    input_folder = args[0]
    output_folder = args[1]

    subs = [x[0] for x in os.walk(input_folder)]

    subs = subs[1:]

    for s in subs:
        if has_subdirectories(s):
            continue

        name = s.replace(input_folder, "")
        print("Preparing %s" % name)
        basename = os.path.basename(s)
        out = (output_folder + name).replace(basename, "")

        prepare_stack.run(s, out)
