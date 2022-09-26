"""
# **************************************************************************#
#                                                                           #
# === General ==============================================================#
#                                                                           #
# **************************************************************************#
# Definition of general helper functionalities
#
# Co-author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
# Co-author: Maximilian Brodbeck <maximilian.brodbeck@isd.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""

import csv
import json
import logging
import os
import shlex
from collections import OrderedDict
from subprocess import check_output
from pathlib import  Path
import gzip
import shutil

# **************************************************************************#
#      Classes                                                              #
# **************************************************************************#
# Definition of Classes


# --------------------------------------------------------------------------#

# **************************************************************************#
#      Functions                                                            #
# **************************************************************************#
# Definition of Functions

# --------------------------------------------------------------------------#

def mkdir_if_not_exist(dir):
    from pathlib import Path
    Path(dir).mkdir(parents=True, exist_ok=True)
    
def set_working_folder(dir: str):
    mkdir_if_not_exist(dir)
    return dir

def splitPath(s: str):
    """
    Splits Filepath into file and path
    usage: file, path = splitPath(s)
    """
    import os
    f = os.path.basename(s)
    p = s[:-(len(f))-1]
    return str(f), str(p)

def get_path_file_extension(input_file):
    file, path = splitPath(input_file)
    file_wo_extension = Path(Path(input_file).stem).stem
    return path, file, file_wo_extension

def load_json(filename):
    """ Load a JSON file
    Args:
        filename (str): Path of a JSON file
    Return:
        Dictionnary of the JSON file
    """
    with open(filename, "r") as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    return data

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def write_txt(filename, lines):
    with open(filename, "a") as f:
        for row in lines:
            f.write("%s\n" % row)

def write_participants(filename, participants):
    with open(filename, "w") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=participants[0].keys())
        writer.writeheader()
        writer.writerows(participants)

def read_participants(filename):
    if not os.path.exists(filename):
        return []
    with open(filename, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row for row in reader]

def splitext_(path, extensions=None):
    """ Split the extension from a pathname
    Handle case with extensions with '.' in it
    Args:
        path (str): A path to split
        extensions (list): List of special extensions
    Returns:
        (root, ext): ext may be empty
    """
    if extensions is None:
        extensions = [".nii.gz"]

    for ext in extensions:
        if path.endswith(ext):
            return path[: -len(ext)], path[-len(ext) :]
    return os.path.splitext(path)

def run_shell_command(commandLine):
    """ Wrapper of subprocess.check_output
    Returns:
        Run command with arguments and return its output
    """
    logger = logging.getLogger(__name__)
    logger.info("Running %s", commandLine)
    return check_output(shlex.split(commandLine))

def ungzip(in_file, out_file):
    with gzip.open(in_file, 'rb') as f_in:
        with open(out_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)