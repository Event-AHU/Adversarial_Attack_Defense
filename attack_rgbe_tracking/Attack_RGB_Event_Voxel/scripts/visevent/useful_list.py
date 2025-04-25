import os
import glob

# 配置路径
root_dir = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/VisEvent/test/"
output_file = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/VisEvent/test/testpair.txt"

def find_aedat4_folders(root_dir, output_file):
    with open(output_file, 'w') as f:
        # 遍历根目录下的所有直接子文件夹
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            
            # 确保是文件夹且不是隐藏文件
            if os.path.isdir(folder_path) and not folder.startswith('.'):
                # 检查文件夹内是否包含.aedat4文件
                aedat_files = glob.glob(os.path.join(folder_path, "*.aedat4"))
                
                # 如果找到则写入绝对路径
                if aedat_files:
                    f.write(f"{(folder)}\n")
                    print(f"Found: {folder}")

if __name__ == "__main__":
    find_aedat4_folders(root_dir, output_file)
    print(f"Results saved to: {output_file}")