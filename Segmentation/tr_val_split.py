# move group bb7f7ca7636d45e9a8cebf3894a0e101 from train to val

import os
import shutil

input_dir = "D:\\segm\\inputs"
output_dir = "D:\\segm\\outputs"
split_ratio = 0.2

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "train"))
    os.makedirs(os.path.join(output_dir, "val"))

groups = {}
for filename in os.listdir(input_dir):
    # extract group ID from filename
    group_id = filename.split("-")[0]
    
    # extract timestamp from filename
    timestamp = int(filename.split("-")[1].split(".")[0])
    
    # add filename and timestamp to the group
    if group_id not in groups:
        groups[group_id] = []
    groups[group_id].append((filename, timestamp))
    
# sort groups by timestamp
groups = sorted(groups.items(), key=lambda x: x[1][-1][1])

# split groups into train and val
train_groups = groups[:int(len(groups) * (1 - split_ratio))]
val_groups = groups[int(len(groups) * (1 - split_ratio)):]

# copy files to the output directory
for group_id, filenames in train_groups:
    avg_timestamp = sum(t[1] for t in filenames) / len(filenames)
    for filename, timestamp in filenames:
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, "train", group_id + "-" + filename.split("-")[1].split(".")[0] + ".png")
        shutil.copyfile(src, dst)

for group_id, filenames in val_groups:
    avg_timestamp = sum(t[1] for t in filenames) / len(filenames)
    for filename, timestamp in filenames:
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, "val", group_id + "-" + filename.split("-")[1].split(".")[0] + ".png")
        shutil.copyfile(src, dst)
