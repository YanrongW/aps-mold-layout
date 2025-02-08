import pandas as pd
import json


def excel_to_json(excel_file_path, json_output_path):
    # 从 Excel 文件加载数据
    xls = pd.ExcelFile(excel_file_path)

    # 遍历每个Sheet表单并转换为JSON
    json_data = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        json_data[sheet_name] = json.loads(df.to_json(orient='records', force_ascii=False))

    # 保存 JSON 数据到文件
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # order_info_path = '../工厂数据_聚合版.xlsx'
    # order_output_path = '../utils/工厂数据.json'
    #
    # fixed_info_path = '../固定信息_国福版.xlsx'
    # fixed_output_path = '../utils/固定信息.json'
    #
    # excel_to_json(order_info_path, order_output_path)
    # excel_to_json(fixed_info_path, fixed_output_path)

    order_info_path = '../utils/产品型号交付限制.xlsx'
    order_output_path = '../utils/产品型号交付限制.json'

    excel_to_json(order_info_path, order_output_path)

