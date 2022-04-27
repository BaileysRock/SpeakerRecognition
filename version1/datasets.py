import os


def gen_file_paths(human_id: str):
    paths: str = []

    for _, dirs, _ in os.walk("./dataset/" + human_id + "/"):
        if len(dirs) == 0:
            continue
        for dir in dirs:
            for path, _, filename in os.walk("./dataset/" + human_id + "/" + dir + "/"):
                for zz in filename:
                    if zz.endswith(".flac"):
                        paths.append(path + zz)

    return paths
