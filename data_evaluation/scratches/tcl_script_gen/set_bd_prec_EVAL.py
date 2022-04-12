"""
copy paste console input into vivado tcl console

"""
precision = 14

# set precision property of all rtl blocks
block_lst = ["machine_0", "nm_0"]
print("\nstartgroup")
for block in block_lst:
    print(f"set_property -dict [list CONFIG.p { {precision} }] [get_bd_cells {block}]")
print("endgroup\n")

# delete ports
in_port_lst = ["cur_data"]
out_port_lst = [("nm_0", "d0_res"), ("nm_0", "d1_res"), ("nm_0", "d2_res")]
print("\nstartgroup")
for in_port in in_port_lst:
    print(f"delete_bd_objs [get_bd_nets {in_port}_1] [get_bd_ports {in_port}]")
for out_port in out_port_lst:
    print(f"delete_bd_objs [get_bd_nets {out_port[0]}_{out_port[1]}] [get_bd_ports {out_port[1]}]")
print("endgroup\n")

# regen ports
block_port_lst_in = [(6*(3+precision)-1, "of1_0", "cur_data")]
block_port_lst_out = [((12+precision)-1, "nm_0", "d0_res"), ((12+precision)-1, "nm_0", "d1_res"),
                      ((12+precision)-1, "nm_0", "d2_res")]
"""
create_bd_port -dir O -from 28 -to 0 d0_res
connect_bd_net [get_bd_pins /nm_0/d0_res] [get_bd_ports d0_res]
"""
print("\nstartgroup")
for p, block, port in block_port_lst_in:
    print(f"create_bd_port -dir I -from {p} -to 0 {port}")
    print(f"connect_bd_net [get_bd_pins /{block}/{port}] [get_bd_ports {port}]")
for p, block, port in block_port_lst_out:
    print(f"create_bd_port -dir O -from {p} -to 0 {port}")
    print(f"connect_bd_net [get_bd_pins /{block}/{port}] [get_bd_ports {port}]")
print("endgroup\n")

print("regenerate_bd_layout")
