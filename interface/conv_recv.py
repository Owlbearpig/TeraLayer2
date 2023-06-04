import matplotlib.pyplot as plt

vals = []
with open("test", "rb") as file:
    data_dump = file.read()
    dump_len = len(data_dump)
    print(data_dump)
    for i in range(len(data_dump)//8):
        data_slice = data_dump[8*i:8*(i+1)]
        #print(data_slice)
        val = int.from_bytes(data_slice, byteorder="little")
        #print(val)
        vals.append(val)

plt.plot(vals)
plt.show()
