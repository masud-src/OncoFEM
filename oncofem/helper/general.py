"""
Definition of general helper functionalities for work with the system.

Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
"""
import logging
import os
import shlex
from subprocess import check_output
from pathlib import Path
import gzip
import shutil

def mkdir_if_not_exist(dir: str, exists_ok=True):
    """
    Makes directory if not exists and returns the string

    *Arguments:*
        dir: String

    *Example:*
        dir = mkdir_if_not_exist(dir) 
    """
    from pathlib import Path
    try:
        Path(dir).mkdir(parents=True, exist_ok=exists_ok)
    except (FileExistsError):
        print("Folder already exists")
    return dir

def splitPath(s: str):
    """
    Splits Filepath into file and path

    *Arguments:*
        s: String

    *Example:*
        file, path = splitPath(s) 
    """
    import os
    f = os.path.basename(s)
    p = s[:-(len(f))-1]
    return str(f), str(p)

def get_path_file_extension(input_file: str):
    """ 
    Returns path, the filename and the filename without extension.

    *Arguments:*
        input_file: String

    *Example:*
        path, file, file_wo_extension = get_path_file_extension(input_file)
    """
    file, path = splitPath(input_file)
    file_wo_extension = Path(Path(input_file).stem).stem
    return path, file, file_wo_extension

def run_shell_command(command: str):
    """ 
    Wrapper of subprocess.check_output. Returns output of that process, can be anything.

    *Arguments:*
        command: String

    *Example:*
        output = run_shell_command(command)
    """
    logger = logging.getLogger(__name__)
    logger.info("Running %s", command)
    return check_output(shlex.split(command))

def file_collector(path: str, ending=None):
    """
    Collects files in folders and subfolders with optional ending.

    *Arguments:*
        path: String
        ending: String (optional)

    *Example:*
        list_of_files = list(file_collector(path, ".nii.gz"))
    """
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if ending is None:
                yield os.path.join(root, filename)
            elif filename.endswith(ending):
                yield os.path.join(root, filename)

def ungzip(in_file: str, out_dir: str):
    """
    Unzips an input file into the output directory.

    *Arguments:*
        in_file: String of input file
        out_file: String of output file
    *Example:*
        ungzip(path, "unzipped_file"))
    """
    with gzip.open(in_file, 'rb') as f_in:
        with open(out_dir, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def check_if_type(var, var_type, return_var):
    """
    Checks if a variable 'var' is of a particular 'var_type'. If yes, the 'return_var' is returned.
    If not, 'var' is returned.

    *Arguments*
        var:        variable of anytype
        var_type:   a particular variable type, i.e. 'str' or 'float'
        return_var: variable that is returned if type of var is var_type

    *Return*
        return_var: variable that is returned if type of var is var_type

    *Example*
        var = check_if_type(var, var_type, return_var)
    """
    if type(var) is var_type:
        return return_var
    else:
        return var

def add_file_appendix(file: str, type="msh"):
    """
    Adds file appendix if it is not set. File type is optional and default 
    is set to "msh". Returns file with appendix.
    """
    if not file.endswith("."+type):
        file += "."+type
    return file
