from snippets.base_converters import twos_compl_to_dec
from pathlib import Path
import re

log_file = "display_output.txt"
vivado_project_path = Path(r"H:\VivadoProjects\project_14\project_14.sim\sim_1\behav\xsim")

log_file_path = ""
glo = vivado_project_path.glob("**/*")
for file in list(glo):
    if log_file in str(file):
        log_file_path = file
        break
with open("converted_file.txt", "w") as outfile:
    with open(log_file_path) as infile:
        lines = infile.readlines()
        for line in lines:

            converted_line, bin_str = "", ""
            for i, char in enumerate(line):
                if char not in ["0", "1"]:
                    if line[i-1] in ["0", "1"]:
                        if (len(bin_str) == 1) | (len(bin_str) == 2): # small bin_strs are likely dec already
                            converted_line += bin_str
                        elif len(bin_str) == 4:
                            bin_str = "0" + bin_str # state is unsigned and 4 bit long number.
                            dec = twos_compl_to_dec(bin_str, p=0)
                            converted_line += str(int(dec))
                        elif len(bin_str) == 29:  # coords
                            dec = twos_compl_to_dec(bin_str, p=17)
                            converted_line += str(round(dec, 3))
                        elif len(bin_str) == 20:  # fx
                            dec = twos_compl_to_dec(bin_str, p=17)
                            converted_line += str(round(dec, 8))
                        elif len(bin_str) < 17:  # counters
                            dec = twos_compl_to_dec(bin_str, p=0)
                            converted_line += str(int(dec))
                        else:
                            converted_line += bin_str
                        bin_str = ""
                    converted_line += char
                else:
                    bin_str += char

            outfile.write(converted_line)
