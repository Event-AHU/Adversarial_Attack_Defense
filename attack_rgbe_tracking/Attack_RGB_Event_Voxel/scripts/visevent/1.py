import os

# 定义文件路径
base_dir = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/VisEvent/train/"
train_voxel_path = os.path.join(base_dir, "train_voxel_list.txt")
trainlist_path = os.path.join(base_dir, "trainlist.txt")
start_frame_idx_path = os.path.join(base_dir, "start_frame_idx.txt")
output_voxel_path = os.path.join(base_dir, "train_voxel_list2.txt")
output_start_idx_path = os.path.join(base_dir, "start_voxel_idx.txt")

# 读取trainlist前500行并建立索引映射
with open(trainlist_path, 'r') as f:
    trainlist_lines = [line.strip() for line in f.readlines()[:500]]
index_map = {line: idx for idx, line in enumerate(trainlist_lines)}

# 读取start_frame_idx全部内容
with open(start_frame_idx_path, 'r') as f:
    start_frame_idx_lines = [line.strip() for line in f.readlines()]

# 处理voxel list并写入结果
with open(train_voxel_path, 'r') as f_in, \
     open(output_voxel_path, 'w') as f_out_voxel, \
     open(output_start_idx_path, 'w') as f_out_start:

    for line in f_in:
        current_line = line.strip()
        if current_line in index_map:
            # 获取对应的索引
            idx = index_map[current_line]
            # 写入voxel list
            f_out_voxel.write(f"{current_line}\n")
            # 写入对应的start frame index
            f_out_start.write(f"{start_frame_idx_lines[idx]}\n")

print("处理完成！结果已保存至：")
print(f"- {output_voxel_path}")
print(f"- {output_start_idx_path}")