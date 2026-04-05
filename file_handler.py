# Read text files as UTF-8.
import os


def read_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("Not a file: " + str(path))
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_multiple(file_paths):
    chunks = []
    for path in file_paths:
        chunks.append(read_file(path))
    return "\n\n".join(chunks)
