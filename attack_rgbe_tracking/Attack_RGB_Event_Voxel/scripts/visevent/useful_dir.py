import os

# 配置路径
root_dir = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/VisEvent/test"
output_file = os.path.join(root_dir, "voxel_list.txt")
def count_files(dir_path):
    """统计目录中的文件数量（排除隐藏文件）"""
    if not os.path.exists(dir_path):
        return -1
    return len([f for f in os.listdir(dir_path) 
                if os.path.isfile(os.path.join(dir_path, f)) and not f.startswith('.')])

def main():
    valid_folders = []
    
    # 遍历根目录下的所有条目
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        
        # 跳过非目录文件和.txt文件
        if not os.path.isdir(entry_path) or entry.endswith(".txt"):
            continue
        
        # 检查两个子目录是否存在
        voxel_dir = os.path.join(entry_path, "voxel")
        vis_imgs_dir = os.path.join(entry_path, "vis_imgs")
        
        # 任意一个子目录不存在则跳过
        if not (os.path.exists(voxel_dir) and os.path.exists(vis_imgs_dir)):
            print(f"[跳过] {entry}: 缺少 voxel 或 vis_imgs 目录")
            continue
        
        # 计算文件数量
        voxel_count = count_files(voxel_dir)
        vis_count = count_files(vis_imgs_dir)
        
        # 输出检查结果
        status = "通过" if voxel_count == vis_count else f"失败 ({voxel_count} vs {vis_count})"
        print(f"[检查] {entry}: {status}")
        
        # 记录有效条目
        if voxel_count == vis_count:
            valid_folders.append(entry)
    
    # 写入结果文件
    with open(output_file, 'w') as f:
        f.write("\n".join(valid_folders))
    
    print(f"\n共找到 {len(valid_folders)} 个有效目录，结果已保存至 {output_file}")

if __name__ == "__main__":
    main()