import os
import sqlite3


def autocomplete(self, text, line, end_idx, dir_only=True):
    """
    https://cmd2.readthedocs.io/en/stable/api/cmd/#cmd2.Cmd.path_complete
    """
    # Determine if a trailing separator should be appended to directory completions
    add_trailing_sep_if_dir: bool = False
    cwd_added: bool = False

    if end_idx == len(line) or (end_idx < len(line) and line[end_idx] != "/"):
        add_trailing_sep_if_dir = True

    # Determine absolute path based on the virtual cwd
    if not text:
        path: str = self.dir
        cwd_added = True

    elif text.startswith("/"):
        path: str = os.path.normpath(text)

    else:
        path: str = os.path.normpath(os.path.join(self.dir, text))
        cwd_added = True

    if text.endswith("/"):
        path += "/"

    base_dir: str = os.path.dirname(path)

    try:
        cursor = self.conn.cursor()
        cursor.execute("""
                    SELECT id FROM filesystem
                    WHERE virtual_path = ? AND is_dir = 1
                """, (base_dir,))
        result = cursor.fetchone()

        if result is None:
            return []

        parent_id = result[0]

        # Get matching children
        cursor.execute("""
            SELECT virtual_path, is_dir FROM filesystem
            WHERE parent_id = ? AND virtual_path LIKE ?
        """, (parent_id, f"{path}%"))
        matches = cursor.fetchall()

        if dir_only:
            matches = [i for i in matches if i[1] == 1]

        if matches:
            # Set this to True for proper quoting of paths with spaces
            self.matches_delimited = True

            # Don't append a space or closing quote to directory
            if len(matches) == 1 and matches[0][1] == 1:
                self.allow_appended_space = False
                self.allow_closing_quote = False

            # Sort the matches before any trailing slashes are added
            matches.sort(key=lambda x: x[0])
            self.matches_sorted = True

            # Build display_matches and add a slash to directories
            for idx, match in enumerate(matches):
                virtual_path, is_dir = match
                self.display_matches.append(virtual_path)
                matches[idx] = virtual_path

                # Add a separator after directories if the next character isn't already a separator
                if is_dir == 1 and add_trailing_sep_if_dir:
                    matches[idx] += "/"
                    self.display_matches[idx] += "/"

            # Remove cwd if it was added to match the text readline expects
            if cwd_added:
                to_replace = self.dir if self.dir == "/" else self.dir + "/"
                matches = [curr_path.replace(to_replace, "", 1) for curr_path in matches]

        return matches

    except sqlite3.Error as e:
        self.perror(f"Autocomplete error: {e}")

        return []
