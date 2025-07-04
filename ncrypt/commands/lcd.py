import os


def do_lcd(self, arg):
    """
    Change the local working directory. Usage: lcd <optional relative or absolute directory path>
    """
    path: str | None = arg.strip()

    if not path or not path.startswith("~"):
        path: str = os.path.normpath(os.path.join(self.ldir, path))

    else:
        path: str = os.path.normpath(os.path.join(os.path.expanduser("~"), path))

    if not os.path.exists(path):
        self.perror(f"No such directory: {path}")

        return

    if not os.path.isdir(path):
        self.perror(f"Not a directory: {path}")

        return

    try:
        os.chdir(path)
        self.ldir = path

    except PermissionError:
        self.perror(f"Permission denied: {path}")


def complete_lcd(self, text, line, start_idx, end_idx):
    return self.path_complete(text, line, start_idx, end_idx, path_filter=os.path.isdir)
