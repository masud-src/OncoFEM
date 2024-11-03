"""
Definition of general helper functionalities for work with the system. It is based on a linux system.

Functions:
    mkdir_if_not_exist:         Makes directory if not exists and returns the string.
    split_path:                 Splits Filepath into file and path
    get_path_file_extension:    Returns path, the filename and the filename without extension.
    run_shell_command:          Wrapper of subprocess.check_output. Returns output of that process, can be anything.
    file_collector:             Collects files in folders and subfolders with optional ending.
    ungzip:                     Unzips an input file into the output directory.
    check_if_type:              Checks if a variable 'var' is of a particular 'var_type'. If yes, the 'return_var' is 
                                returned. If not, 'var' is returned.
    add_file_appendix:          Adds file appendix if it is not set. File type is optional and default is set to "msh". 
                                Returns file with appendix.
"""
from typing import Union, Generator, Any
import os
import subprocess
import shlex
from pathlib import Path
import gzip
import shutil

def mkdir_if_not_exist(directory: str) -> str:
    """
    Makes directory if not exists and returns the string

    *Arguments*:
        dir: String

    *Example*:
        dir = mkdir_if_not_exist(dir) 
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def split_path(s: str) -> tuple[str, str]:
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

def get_path_file_extension(input_file: str) -> tuple[str, str, str]:
    """ 
    Returns path, the filename and the filename without extension.

    *Arguments:*
        input_file: String

    *Example:*
        path, file, file_wo_extension = get_path_file_extension(input_file)
    """
    file, path = split_path(input_file)
    file_wo_extension = Path(Path(input_file).stem).stem
    return path, file, file_wo_extension

def run_shell_command(command: str) -> Union[subprocess.CompletedProcess, subprocess.CompletedProcess[bytes]]:
    """ 
    Wrapper of subprocess.check_output. Returns output of that process, can be anything.

    *Arguments:*
        command: String

    *Example:*
        output = run_shell_command(command)
    """
    return subprocess.run(shlex.split(command))

def file_collector(path: str, ending:str=None) -> Generator[str, Any, None]:
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

def ungzip(in_file: str, out_dir: str) -> None:
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

def check_if_type(var: Any, var_type: Any, return_var: Any) -> Any:
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

def add_file_appendix(file: str, type:str="msh") -> str:
    """
    Adds file appendix if it is not set. File type is optional and default 
    is set to "msh". Returns file with appendix.

    *Arguments*
        file:       String of input file
        type:       String of appendix

    *Return*
        file:       String of file with appendix

    *Example*:
        var = add_file_appendix("brain_file"):
    """
    if not file.endswith("."+type):
        file += "."+type
    return file
