import os


def do_lmkdir(self, arg):
    """
    Create a local directory. Usage: lmkdir <relative or absolute directory path>
    """
    path: str = arg.strip()

    if not path:
        self.perror("Missing directory path")

        return

    if not path.startswith("~"):
        path: str = os.path.normpath(os.path.join(self.ldir, path))

    else:
        path: str = os.path.normpath(os.path.join(os.path.expanduser("~"), path))

    try:
        os.makedirs(path, exist_ok=False)

    except FileExistsError:
        self.perror(f"Directory already exists: {path}")

        return

    except PermissionError:
        self.perror(f"Permission denied: {path}")

        return

    except Exception as e:
        self.perror(f"Error creating directory {path}: {e}")
