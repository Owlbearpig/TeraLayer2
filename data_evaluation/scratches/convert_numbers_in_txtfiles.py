from snippets.base_converters import twos_compl_to_dec
from pathlib import Path
import os

log_file_stub = "sim_output"
#log_file_stub = "display_output_mult"
#log_file_stub = "display_output_loss_after"
#log_file_stub = "display_output_lut_fp_division"

if os.name == "posix":
    vivado_project_path = Path(r"/media/alex/WDElements/IPs")
else:
    vivado_project_path = Path(r"H:\IPs")
p = 8
show_bin_str = False


def convert_file(file_path):
    outfile_name = "converted_" + str(file_path.name)
    print(f"Converting: {str(file_path)}. \nConverted file name: {outfile_name}")
    with open(outfile_name, "w") as outfile:
        with open(file_path) as infile:
            lines = infile.readlines()

            for line in lines:
                converted_line, bin_str = "", ""
                for i, char in enumerate(line):
                    if char not in ["0", "1"]:
                        if line[i - 1] in ["0", "1"]:

                            if (len(bin_str) == 1) | (len(bin_str) == 2):  # small bin_strs are likely dec already
                                converted_line += bin_str
                            elif len(bin_str) == 8:
                                bin_str = "0" + bin_str  # 8 bit cntr
                                dec = twos_compl_to_dec(bin_str, p=0)
                                converted_line += str(int(dec))
                            elif len(bin_str) == 5:
                                bin_str = "0" + bin_str  # state is unsigned and 4 bit long number.
                                dec = twos_compl_to_dec(bin_str, p=0)
                                # converted_line += f"{str(int(dec))}, ({bin_str[1:]})"
                                converted_line += f"{str(int(dec))}"
                            elif len(bin_str) == 2 + 14:
                                dec = twos_compl_to_dec(bin_str, p=14)
                                converted_line += str(round(dec, 6))
                            elif len(bin_str) == 2 * 3 + 2 * p:
                                dec = twos_compl_to_dec(bin_str, p=2*p)
                                converted_line += str(round(dec, 6))
                            elif len(bin_str) == 12 + p:  # coords
                                dec = twos_compl_to_dec(bin_str, p=p)
                                converted_line += str(round(dec, 3))
                            elif len(bin_str) == 3 + p:  # fx
                                dec = twos_compl_to_dec(bin_str, p=p)
                                converted_line += str(round(dec, 8))
                            else:
                                converted_line += bin_str
                            if show_bin_str:
                                converted_line += f" ({bin_str})"
                            bin_str = ""
                        converted_line += char
                    else:
                        bin_str += char

                outfile.write(converted_line)


glo = vivado_project_path.glob("**/*")

for file in list(glo):
    if log_file_stub in str(file.name):
        convert_file(file)
