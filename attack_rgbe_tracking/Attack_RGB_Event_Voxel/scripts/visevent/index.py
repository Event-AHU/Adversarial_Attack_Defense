import os

# 配置路径
base_dir = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/VisEvent/train"
train_list_path = os.path.join(base_dir, "trainlist.txt")
start_frame_path = os.path.join(base_dir, "start_frame_idx.txt")
train_voxel_list_path = os.path.join(base_dir, "train_voxel_list.txt")
output_idx_path = os.path.join(base_dir, "start_voxel_idx.txt")
output_list_path = os.path.join(base_dir, "train_voxel_list2.txt")  # 新增输出文件

def main():
    # 读取trainlist.txt建立索引字典（值 -> 行号）
    with open(train_list_path, 'r') as f:
        trainlist = [line.strip() for line in f.readlines()]
    value_to_linenum = {value: idx for idx, value in enumerate(trainlist, start=1)}

    # 读取start_frame数据
    with open(start_frame_path, 'r') as f:
        start_frames = [line.strip() for line in f.readlines()]

    # 初始化计数器
    valid_count = 0
    skipped_count = 0

    # 处理voxel列表（同时写入两个文件）
    with open(train_voxel_list_path, 'r') as fin, \
         open(output_idx_path, 'w') as fout_idx, \
         open(output_list_path, 'w') as fout_list:
        
        for line_num, value in enumerate(fin, start=1):
            value = value.strip()
            
            # ------------------------------
            # 阶段1：验证trainlist是否存在
            # ------------------------------
            if value not in value_to_linenum:
                print(f"[警告] 跳过：'{value}' 在 trainlist.txt 中不存在 (第 {line_num} 行)")
                skipped_count += 1
                continue
            
            # ------------------------------
            # 阶段2：验证start_frame范围
            # ------------------------------
            frame_linenum = value_to_linenum[value] - 1  # 转换为0-based索引
            if frame_linenum >= len(start_frames):
                print(f"[警告] 跳过：start_frame_idx.txt 没有第 {value_to_linenum[value]} 行数据")
                skipped_count += 1
                continue
            
            # ------------------------------
            # 有效数据写入
            # ------------------------------
            try:
                # 写入start_voxel_idx.txt
                fout_idx.write(f"{start_frames[frame_linenum]}\n")
                # 写入train_voxel_list2.txt
                fout_list.write(f"{value}\n")
                valid_count += 1
            except Exception as e:
                print(f"[错误] 写入失败：{str(e)}")
                skipped_count += 1

    # 输出统计信息
    print(f"\n处理完成：")
    print(f"成功匹配条目：{valid_count}")
    print(f"跳过无效条目：{skipped_count}")
    print(f"start_voxel_idx.txt 已生成：{output_idx_path}")
    print(f"train_voxel_list2.txt 已生成：{output_list_path}")

if __name__ == "__main__":
    main()