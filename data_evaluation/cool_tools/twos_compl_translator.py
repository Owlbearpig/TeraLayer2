import PySimpleGUI as sg
from scratches.snippets.base_converters import convert_lines, dec_to_twoscompl, twos_compl_to_dec


def translate_string(s, p):
    p = int(p)

    return "\n".join(convert_lines(s, p))

def translate_single_string(s, p):
    p = int(p)
    return twos_compl_to_dec(s, p)


def conv_to_verilog(s, p, pd):
    # Garbage ): Idd...
    p, pd = int(p), int(pd)
    ret_lines = []

    list_elems = s.split(" ")

    list_is_complex = any(["j" in elem for elem in list_elems])
    if list_is_complex:
        ret_lines.append("real_parts\n")
        for elem in list_elems:
            ret_line = ""
            elem_clean = ''.join(c for c in elem if c.isdigit() or (c in [".", "j", "+", "-"]))
            if len(elem_clean) > 0:
                number = complex(elem_clean)
            else:
                continue
            bit_s = f"{dec_to_twoscompl(number.real, pd, p, format=True)}"
            ret_line += bit_s + f", // {number}\n"
            ret_lines.append(ret_line)
        ret_lines.append("imag_parts\n")
        for elem in list_elems:
            ret_line = ""
            elem_clean = ''.join(c for c in elem if c.isdigit() or (c in [".", "j", "+", "-"]))
            if len(elem_clean) > 0:
                number = complex(elem_clean)
            else:
                continue
            bit_s = f"{dec_to_twoscompl(number.imag, pd, p, format=True)}"
            ret_line += bit_s + f", // {number}\n"
            ret_lines.append(ret_line)
    else:
        for elem in list_elems:
            ret_line = ""
            elem_clean = ''.join(c for c in elem if c.isdigit() or (c in [".", "j", "+", "-"]))
            if len(elem_clean) > 0:
                number = float(elem_clean)
            else:
                continue
            bit_s = f"{dec_to_twoscompl(number, pd, p, format=True)}"
            ret_line += bit_s + f", // {number}\n"
            ret_lines.append(ret_line)

    return "".join(ret_lines)


sg.theme('SandyBeach')

layout = [
    [sg.Text("precision"), sg.Input(key="precision", default_text='22', size=(15, 1)),
     sg.Text("integer precision"), sg.Input(key="int_prec", default_text='3', size=(15, 1)), ],
    [sg.Text("Twos Complement String"), sg.InputText(key="single_line_base2", size=(15, 1)),
     sg.Text("Decimal String"), sg.InputText(key="single_line_base10", size=(15, 1))],
    [sg.Text("Base 2s complement text")],
    [sg.Multiline(key="textfrom", size=(100, 20)), sg.Multiline(key="texto", size=(100, 20))],
    [sg.Button("Translate", key="Translate"), sg.Button("List to .v", key="convVerilog")]
]

window = sg.Window('Simple data entry window', layout)


def event_handler(event, values):
    if event == "Translate":
        if values["textfrom"]:
            converted_str = translate_string(s=values["textfrom"], p=values["precision"])
            window["texto"].update(converted_str)

        base10_str = translate_single_string(s=values["single_line_base2"], p=values["precision"])
        window["single_line_base10"].update(base10_str)
    elif event == "convVerilog":
        converted_str = conv_to_verilog(s=values["textfrom"], p=values["precision"], pd=values["int_prec"])
        window["texto"].update(converted_str)
    else:
        return


while True:
    event, values = window.read()
    event_handler(event, values)

    if event == sg.WINDOW_CLOSED:
        break

window.close()

# The input data looks like a simple list
# when automatic numbered
