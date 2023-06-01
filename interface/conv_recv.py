import matplotlib.pyplot as plt

vals = []
with open("test", "rb") as file:
    lines = file.readlines()

    for line in lines:
        b = line
        print(b)
        try:
            val = int.from_bytes(eval(b), byteorder="little")
            print(val)
            vals.append(val)
        except ValueError:
            continue

plt.plot(vals)
plt.show()
