import os

path = "./raw"
test_lst = os.listdir(path)

test_file = open("test.txt", "w")
for img in test_lst:
    test_file.write(img.split('.jpg')[0] + '\n')