import pandas as pd
import json


def json_to_dataframes(json_file_path):
    # 从 JSON 文件加载数据
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    # 遍历每个Sheet表单的JSON数据并构建数据框
    dataframes = {}
    for sheet_name, sheet_data in json_data.items():
        # 从JSON数据创建数据框
        df = pd.read_json(sheet_data, orient='records')
        # 将数据框存储到字典中
        dataframes[sheet_name] = df

    return dataframes

# 示例用法
order_info_path = '../mold_layout/工厂数据.json'
fixed_info_path = '../mold_layout/固定信息.json'
order_info_dataframes = json_to_dataframes(order_info_path)
print(order_info_dataframes['模具上周在线状态'])
fixed_info_dataframes = json_to_dataframes(fixed_info_path)
print(fixed_info_dataframes)

# 现在，excel_dataframes 是一个字典，其中每个键是Sheet表单的名称，对应的值是该表单的数据框
# 可以通过 excel_dataframes['Sheet1'] 访问特定的数据框