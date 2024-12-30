import json


def count_category_instances(annotation_file):
    """
    统计COCO数据集中每个类别的目标数量。

    参数:
        annotation_file (str): COCO格式的注解文件路径

    返回:
        dict: 每个类别的实例数量
    """
    # 读取注解文件
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 提取类别信息
    categories = {category['id']: category['name'] for category in coco_data['categories']}

    # 初始化类别计数器
    category_counts = {category['name']: 0 for category in coco_data['categories']}

    # 遍历注释并统计每个类别的实例数量
    for annotation in coco_data['annotations']:
        category_id = annotation['category_id']
        if category_id in categories:
            category_name = categories[category_id]
            category_counts[category_name] += 1

    return category_counts


# 使用示例
annotation_file_path = 'constructionsitesafety_dataset_yolo/cssd_valid.json'  # 替换为你的COCO注解文件路径
category_counts = count_category_instances(annotation_file_path)

# 打印结果
for category, count in category_counts.items():
    print(f"Category: {category}, Count: {count}")