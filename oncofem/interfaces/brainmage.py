"""
# **************************************************************************#
#                                                                           #
# === BrainMage module  ====================================================#
#                                                                           #
# **************************************************************************#
# In this module an interface to the brain mage package is implemented.
# With this the user can perform skull stripping.
# 
#
# Author: Marlon Suditsch <marlon.suditsch@mechbau.uni-stuttgart.de>
#
# --------------------------------------------------------------------------#
"""
#TODO: create interface for training of own data
import subprocess
from oncofem.struc.state import State

class BrainMaGe:

    def __init__(self):
        self.dev = "cpu"

    def single_run(self, input_file: str, output_file: str, mask_file: str):
        command = ["brain_mage_single_run"]
        command.append("-i")
        command.append(input_file)
        command.append("-o")
        command.append(output_file)
        command.append("-m")
        command.append(mask_file)
        command.append("-dev")
        command.append(self.dev)
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        print(p.communicate())

    def multi_4_run(self, input_files: list, output: str):
        command = ["brain_mage_single_run_multi_4"]
        command.append("-i")
        command.append(input_files[0])
        command.append("-i")
        command.append(input_files[1])
        command.append("-i")
        command.append(input_files[2])
        command.append("-i")
        command.append(input_files[3])
        command.append("-o")
        command.append(output)
        command.append("-dev")
        command.append(self.dev)
        print(command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        print(p.communicate())

