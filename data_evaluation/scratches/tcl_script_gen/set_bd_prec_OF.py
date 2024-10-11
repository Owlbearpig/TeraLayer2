"""
copy paste console input into vivado tcl console

"""
precision = 10

# set precision property of all rtl blocks
block_lst = ["controller_0", "input_module_0", "cordic_format_0", "cordic_format_1", "cordic_format_2",
             "cordic_format_3", "cos_0", "cos_1", "cos_2", "cos_3", "sin_0", "sin_1", "sin_2", "sin_3",
             "combine_0", "one_cycle_buffer_0", "lut_fp_division_0", "fp_multiplication_0", "loss_0"]
print("\nstartgroup")
for block in block_lst:
    print(f"set_property -dict [list CONFIG.p { {precision} }] [get_bd_cells {block}]")
print("endgroup\n")

# delete ports
in_port_lst = ["i_d2", "i_d1", "i_d0", "cur_data"]
out_port_lst = [("loss_0", "o_total_loss")]
print("\nstartgroup")
for in_port in in_port_lst:
    print(f"delete_bd_objs [get_bd_nets {in_port}_1] [get_bd_ports {in_port}]")
for out_port in out_port_lst:
    print(f"delete_bd_objs [get_bd_nets {out_port[0]}_{out_port[1]}] [get_bd_ports {out_port[1]}]")
print("endgroup\n")

# recreate ports
block_port_lst_in = [(12+precision-1, "controller_0", "i_d0"), (12+precision-1, "controller_0", "i_d1"),
                  (12+precision-1, "controller_0", "i_d2"), (6*(3+precision)-1, "loss_0", "cur_data")]
block_port_lst_out = [((3+precision)-1, "loss_0", "o_total_loss")]

print("\nstartgroup")
for p, block, port in block_port_lst_in:
    print(f"create_bd_port -dir I -from {p} -to 0 {port}")
    print(f"connect_bd_net [get_bd_pins /{block}/{port}] [get_bd_ports {port}]")
for p, block, port in block_port_lst_out:
    print(f"create_bd_port -dir O -from {p} -to 0 {port}")
    print(f"connect_bd_net [get_bd_pins /{block}/{port}] [get_bd_ports {port}]")
print("endgroup\n")

print("regenerate_bd_layout")
