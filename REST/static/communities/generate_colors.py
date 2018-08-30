import random, json

def getRandomColor():
    letters = '0123456789ABCDEF'
    return "#" + "".join([random.choice(letters) for i in range(6)])


colors = [getRandomColor() for i in range(100000)]


with open("colors.json", "w") as f:
    json.dump(colors, f)