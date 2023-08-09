import matplotlib.pyplot as plt
# from bitstring import BitArray
import binascii
chunk = 8

vals = []
with open("dump", "rb") as file:
    #data_dump = file.read()
    #print(data_dump[0:20])
    #s = data_dump[0:20]
    #c = BitArray(hex=str(s))
    #print(c.bin)

    data_dump = file.read()
    dump_len = len(data_dump)
    #print(data_dump)
    #print("".join(str(data_dump).split(r"\x")))
    for i in range(len(data_dump)//chunk):
        data_slices = binascii.hexlify(data_dump[i * chunk:(i + 1) * chunk]).decode()
        #print(data_slices)
        s = [data_slices[i:i + 2] for i in range(0, len(data_slices), 2)]
        #s = [c[1] + c[0] for c in s]
        #print(s)
        input_str = '0x' + "".join(list(reversed(s)))
        val = int(input_str, 16)
        vals.append(val)
        if i < 300:
            print(i, s, int(input_str, 16))
        #input_str = '0x' + "".join(list(reversed(s)))
        #print(input_str)

        #print()
        #val = int.from_bytes(data_dump[i * chunk:(i + 1) * chunk], byteorder="little")
        #print(val)
        #vals.append(val)

plt.plot(vals)
plt.show()
