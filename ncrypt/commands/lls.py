import os
import stat
from datetime import datetime


def do_lls(self, arg):
    """
    List contents of the local directory. Usage: lls <optional relative or absolute directory path>
    """
    path: str | None = arg.strip()

    if not path or not path.startswith("~"):
        path: str = os.path.normpath(os.path.join(self.ldir, path))

    else:
        path: str = os.path.normpath(os.path.join(os.path.expanduser("~"), path))

    try:
        entries: list[str] = sorted(os.listdir(path))

    except FileNotFoundError:
        self.perror(f"No such directory: {path}")

        return

    except PermissionError:
        self.perror(f"Permission denied: {path}")

        return

    self.poutput(f"Contents of {path}:\n")

    for entry in entries:
        full_path = os.path.join(path, entry)

        try:
            st = os.stat(full_path)
            perms = stat.filemode(st.st_mode)
            size = st.st_size
            modified_time = datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M')
            entry_type = "/" if os.path.isdir(full_path) else ""
            self.poutput(f"{perms} {size:>10} {modified_time} {entry}{entry_type}")

        except Exception as e:
            self.perror(f"Error reading {entry}: {e}")


def complete_lls(self, text, line, start_idx, end_idx):
    return self.path_complete(text, line, start_idx, end_idx, path_filter=os.path.isdir)
