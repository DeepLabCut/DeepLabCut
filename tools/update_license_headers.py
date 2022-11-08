"""Apply copyright headers to all code files in the repository.

This file can be called as a python script without arguments. For
configuration, see the instructions in NOTICE.yml.
"""

import tempfile
import glob
import yaml
import fnmatch
import subprocess


def load_config(filename):
    with open(filename, "r") as fh:
        config = yaml.safe_load(fh)
    return config


def walk_directory(entry):
    """Talk the directory"""

    if "header" not in entry:
        raise ValueError("Current entry does not have a header.")
    if "include" not in entry:
        raise ValueError("Current entry does not have an include list.")

    def _list_include():
        """List all files specified in the include list."""
        for include_pattern in entry["include"]:
            for filename in glob.iglob(include_pattern, recursive=True):
                yield filename

    def _filter_exclude(iterable):
        """Filter filenames from an iterator by the exclude patterns."""
        for filename in iterable:
            for exclude_pattern in entry.get("exclude", []):
                if fnmatch.fnmatch(filename, exclude_pattern):
                    break
            else:
                yield filename

    files = _filter_exclude(set(_list_include()))
    return list(files)


def main(input_file="NOTICE.yml"):
    config = load_config(input_file)
    for entry in config:
        filelist = list(walk_directory(entry))
        with tempfile.NamedTemporaryFile(mode="w") as header_file:
            header_file.write(entry["header"])
            header_file.flush()
            header_file.seek(0)
            command = ["licenseheaders", "-t", str(header_file.name), "-f"] + filelist
            result = subprocess.run(command, capture_output=True)
            if result.returncode != 0:
                print(result.stdout.decode())
                print(result.stderr.decode())


if __name__ == "__main__":
    main()
