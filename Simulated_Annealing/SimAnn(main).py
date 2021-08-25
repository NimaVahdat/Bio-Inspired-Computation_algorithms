from SimAnnTSP import SimAnneal

flag = False
coords = []
with open("bayg29.tsp", "r") as f:
    for line in f.readlines():
        line = line.split()
        if line == ["EOF"]:
            flag = False
        if flag:
            line0 = [float(x.replace("\n", "")) for x in line]
            coords.append(line0[1:])
        if line == ["DISPLAY_DATA_SECTION"]:
            flag = True

print(coords)

if __name__ == "__main__":
    sa = SimAnneal(coords, stopping_iter=5000, alpha = 0.95)
    sa.batch_anneal()