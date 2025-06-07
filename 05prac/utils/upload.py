import os


def upload_file(file):
    file_path = f"./.cache/files/{file.name}"
    os.makedirs(".cache", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path
