import os

file_path = "/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c01/DATA/dataset/VisEvent/test/voxel_train.txt"

try:
    # 自动创建目录（如果不存在）
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 生成并写入数字序列
    with open(file_path, 'w', encoding='utf-8') as f:
        for number in range(201):
            f.write(f"{number}\n")  # 显式添加换行符
    
    # 验证写入结果
    with open(file_path, 'r') as f:
        line_count = sum(1 for _ in f)
    
    print(f"文件已生成至：{file_path}")
    print(f"理论行数：267行（0到266）")
    print(f"实际验证行数：{line_count}行") 

except Exception as e:
    print(f"生成文件时出现错误：{str(e)}")
    print("可能原因：")
    print("1. 磁盘空间不足")
    print("2. 路径权限不足")
    print("3. 路径包含非法字符")