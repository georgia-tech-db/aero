from sys import argv


def extract(line):
    res_list = []
    line_list = line.split(" ")

    start_append = False
    for item in line_list:
        if item == "jcplan":
            start_append = not start_append
        else:
            if start_append:
                res_list.append(item)

    return " ".join(res_list)


path = argv[1]

data_list = []
for line in open(path).read().splitlines():
    if "jcplan" in line:
        data = extract(line)
        id, c1, c2 = data.split(",")
        id = int(id)
        c1 = float(c1)
        c2 = float(c2)
        data_list.append((id, c1, c2))

data_list = sorted(data_list, key=lambda item: item[0])

for data in data_list:
    data = [str(d) for d in data]
    print(",".join(data))