from snippets.base_converters import twos_compl_to_dec
from pathlib import Path
import re

log_file_stub = "sim_output"
vivado_project_path = Path(r"/media/alex/WDElements/IPs/eval")


def convert_file(file_path):
    print("convert:", file_path)
    with open("converted" + str(file_path.name).replace("sim_output", "sim_output"), "w") as outfile:
        with open(file_path) as infile:
            lines = infile.readlines()
            for line in lines:

                converted_line, bin_str = "", ""
                for i, char in enumerate(line):
                    if char not in ["0", "1"]:
                        if line[i-1] in ["0", "1"]:
                            p = 17
                            if (len(bin_str) == 1) | (len(bin_str) == 2): # small bin_strs are likely dec already
                                converted_line += bin_str
                            elif len(bin_str) == 8:
                                bin_str = "0" + bin_str # 8 bit cntr
                                dec = twos_compl_to_dec(bin_str, p=0)
                                converted_line += str(int(dec))
                            elif len(bin_str) == 4:
                                bin_str = "0" + bin_str # state is unsigned and 4 bit long number.
                                dec = twos_compl_to_dec(bin_str, p=0)
                                converted_line += str(int(dec))
                            elif len(bin_str) == 34:
                                dec = twos_compl_to_dec(bin_str, p=22)
                                converted_line += str(round(dec, 6))
                            elif len(bin_str) == 25:
                                dec = twos_compl_to_dec(bin_str, p=22)
                                converted_line += str(round(dec, 6))
                            elif len(bin_str) == 29:  # coords
                                dec = twos_compl_to_dec(bin_str, p=p)
                                converted_line += str(round(dec, 3))
                            elif len(bin_str) == 20:  # fx
                                dec = twos_compl_to_dec(bin_str, p=p)
                                converted_line += str(round(dec, 8))
                            elif len(bin_str) == 50: # input module, m
                                dec = twos_compl_to_dec(bin_str, p=20+p)
                                converted_line += str(round(dec, 8))
                            elif len(bin_str) == 23: # cordic format, r_int
                                dec = twos_compl_to_dec(bin_str, p=p)
                                converted_line += str(round(dec, 8))
                            elif len(bin_str) == 44: # cordic format, m
                                dec = twos_compl_to_dec(bin_str, p=2*p)
                            elif len(bin_str) == 45: # cordic format, r
                                dec = twos_compl_to_dec(bin_str, p=2*p)
                                converted_line += str(round(dec, 8))
                            elif len(bin_str) == 21: # cos, sin, c
                                dec = twos_compl_to_dec(bin_str, p=p)
                                converted_line += str(round(dec, 8))
                            elif len(bin_str) == 37: # cos, sin, m1
                                dec = twos_compl_to_dec(bin_str, p=2*p)
                                converted_line += str(round(dec, 8))
                            elif len(bin_str) < 17:  # counters
                                dec = twos_compl_to_dec(bin_str, p=0)
                                converted_line += str(int(dec))
                            elif len(bin_str) == 40:  # multiplied numbers (e.g. 3_17 * 3_17)
                                dec = twos_compl_to_dec(bin_str, p=2*p)
                                converted_line += str(round(dec, 8))
                            else:
                                converted_line += bin_str
                            bin_str = ""
                        converted_line += char
                    else:
                        bin_str += char

                outfile.write(converted_line)


glo = vivado_project_path.glob("**/*")
for file in list(glo):
    if log_file_stub in str(file.name):
        convert_file(file)