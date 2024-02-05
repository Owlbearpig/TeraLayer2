from pathlib import Path
from scipy.constants import c as c0
from numpy import pi
import os
from os import name as os_name

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

if 'posix' in os_name:
    base_dir = Path(r"/home/ftpuser/ftp/Data/")
    data_dir = base_dir / "TeraLayer2" / "Foil_glue" / "Img0"
    # result_dir = Path(r"/home/alex/MEGA/AG/Projects/TeraLayer/Implementation/Results")
    result_dir = Path(r"/home/alex/MEGA/AG/Projects/TeraLayer/Endreport/Figures")
else:
    data_dir = Path(r"")
    # result_dir = Path(r"E:\Mega\AG\Projects\TeraLayer\Implementation\Results")
    result_dir = Path(r"E:\Mega\AG\Projects\TeraLayer\Endreport\Figures")

try:
    os.scandir(data_dir)
except FileNotFoundError as e:
    raise e

post_process_config = {"sub_offset": True, "en_windowing": False}

# physical constants
THz = 1e12
c_thz = c0 * 1e-6  # um / ps -> 1e6 / 1e-12 = 1e-6
thea = 8 * pi / 180

# plotting
plot_range = slice(25, 250)
plot_range1 = slice(5, 450)
plot_range2 = slice(25, 250)
