import json
import csv
import os
dirs = []
files = []
def traversal_files(path):
    for item in os.scandir(path):
        if item.is_dir():
            dirs.append(item.name)
        elif item.is_file():
            files.append(item.name)

    print('dirs:', dirs)
    print('files:', files)


os_path= '/public/data/data/imagenet/Sample_1000/'
path = 'transfer_attack/imagenet_class_index.json'
# with open(path, "r") as f:
#     row_data = json.load(f)
# # 读取每一条json数据

# for cls in row_data.items():
#     print(cls[1][0])

# for d in row_data:
#     print(d)
traversal_files(os_path)