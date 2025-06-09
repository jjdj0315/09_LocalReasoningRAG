import os


def create_dir():
    if not os.path.exists(".cache"):
        os.mkdir(".cache")

    if not os.path.exists(".cache/embeddings"):
        os.mkdir(".cache/embeddings")

    if not os.path.exists(".cache/files"):
        os.mkdir(".cache/files")
