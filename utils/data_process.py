import json
import pandas as pd
import numpy as np

def json_to_dataframes(json_data):
    json_data = json.loads(json_data)
    # 遍历每个Sheet表单的JSON数据并构建数据框
    dataframes = {}
    for sheet_name in json_data.keys():
        # 从JSON数据创建数据框
        df = pd.DataFrame(json_data[sheet_name])
        # 将数据框存储到字典中
        dataframes[sheet_name] = df

    return dataframes


def process_factory_data(order_info):
    # 处理输入订单数据
    input_dict = json_to_dataframes(order_info)
    print(input_dict)

    # 待生产订单
    all_order_df = input_dict['orders_to_be_produced']
    all_order_df.columns = ['mould_series', 'product_type', 'delivery_time', 'order_quantity']
    all_order_df = all_order_df[all_order_df.order_quantity > 0]

    # 上周在线模具状态
    mould_online_state = input_dict['mould_online_state_last_week']
    mould_online_state.columns = ['mould_series', 'online_mould_cnt', 'online_bus_code']

    return all_order_df, mould_online_state


def load_fixed_information(fixed_info):
    input_dict = json_to_dataframes(fixed_info)
    print(input_dict)

    # 模具信息
    mould_info = input_dict['mould_info']
    mould_info.columns = ['mould_series', 'mould_code', 'mould_cnt', 'bus_code', 'unit_capacity']
    # 建立字典映射
    row_mark = pd.unique(mould_info['mould_code'])
    mould_code_mapping = {code: 'm' + str(i) for i, code in enumerate(row_mark)}
    mould_info['mould_code'] = mould_info['mould_code'].map(mould_code_mapping)
    # with open(mapping_file_name, 'w') as json_file:
    #     json.dump(mould_code_mapping, json_file, indent=4)
    mould_info['total_capacity'] = mould_info.mould_cnt * mould_info.unit_capacity

    # 模具与线体的约束关系
    mould_to_bus = {}
    for _, row in mould_info[['mould_code', 'bus_code']].iterrows():
        mould_to_bus[row['mould_code']] = row['bus_code']

    # 模具对应型号
    mould_mapping_product = input_dict['mould_mapping_product']
    mould_mapping_product.columns = ['product_code', 'mould_series']
    mould_mapping_product = pd.merge(mould_mapping_product, mould_info[['mould_series', 'mould_code']],
                                     on='mould_series', copy=False)

    # 线体信息
    bus_info = input_dict['bus_info']
    bus_info.columns = ['bus_code', 'capacity_per_day', 'bus_name', 'fixture_cnt']

    bus_dict = {}
    for _, row in bus_info.iterrows():
        bus_dict[row['bus_code']] = [row['capacity_per_day'], row['fixture_cnt']]

    return mould_info, mould_mapping_product, mould_to_bus, bus_info, bus_dict, mould_code_mapping


order_info_file = "../utils/工厂数据.json"
with open(order_info_file, 'r', encoding='utf-8') as order_file:
    order_info = order_file.read()

fixed_info_file = "../utils/固定信息.json"
with open(fixed_info_file, 'r', encoding='utf-8') as fixed_file:
    fixed_info = fixed_file.read()

df_orders_to_be_produced, mould_online_state = process_factory_data(order_info)
df_orders_to_be_produced['delivery_time'] = pd.to_datetime(df_orders_to_be_produced['delivery_time'])

mould_online_state = mould_online_state.drop(columns=['online_bus_code'])
print(mould_online_state)

df_orders_to_be_produced['delivery_time'] = pd.to_datetime(df_orders_to_be_produced['delivery_time'])

# # 设置时间范围
# date_range = pd.date_range(start="2023-12-04", end="2023-12-09", freq='D')
#
# # 遍历日期范围，创建新列并填入订单量
# for date in date_range:
#     day_name = date.strftime('%A').lower()  # 获取星期几的字符串，例如 'friday'
#     day_quantity_column = f"{day_name}_delivery_quantity"
#
#     # 在 orders_to_be_produced 中添加新列，初始化为 0
#     df_orders_to_be_produced[day_quantity_column] = 0
#
#     # 将对应日期的订单量填入新列
#     mask = df_orders_to_be_produced['delivery_time'] == date
#     df_orders_to_be_produced.loc[mask, day_quantity_column] = df_orders_to_be_produced.loc[mask, 'order_quantity']
df_orders_to_be_produced = df_orders_to_be_produced.drop(columns=['product_type', 'delivery_time'])
df_orders_to_be_produced['order_quantity'] = df_orders_to_be_produced.groupby('mould_series')['order_quantity'].transform('sum')
df_orders_to_be_produced = df_orders_to_be_produced.drop_duplicates()
df_orders_to_be_produced = df_orders_to_be_produced.reset_index(drop=True)
# 最终的结果 DataFrame
print(df_orders_to_be_produced)


mould_info, mould_mapping_product, mould_to_bus, bus_info, bus_dict, mould_code_mapping = load_fixed_information(fixed_info)

mould_mapping = pd.merge(df_orders_to_be_produced, mould_info[['mould_series', 'mould_cnt', 'unit_capacity']],
                                     on='mould_series', copy=False)

mould_mapping['day_quantity'] = np.ceil(mould_mapping['order_quantity'] / mould_mapping['unit_capacity']).astype(int)
merged_data = {
    "mould_online_state_last_week": mould_online_state.to_dict(orient='records'),
    "mould_configs": mould_mapping.drop(columns=['order_quantity']).to_dict(orient='records'),
    "max_fixture_count": 18
}

# 将合并后的数据写入 JSON 文件
with open('模具配置.json', 'w', encoding='utf-8') as json_file:
    json.dump(merged_data, json_file, indent=4, ensure_ascii=False)

mould_mapping['order_quantity'] = mould_mapping['day_quantity'] * mould_mapping['unit_capacity']
mould_mapping = mould_mapping.drop(columns=['order_quantity'])
print(mould_mapping)

# data = {
#     'mould_series': ['Series1', 'Series1', 'Series2', 'Series2', 'Series3', 'Series3'],
#     'product_code': ['Prod1', 'Prod2', 'Prod3', 'Prod4', 'Prod5', 'Prod6'],
#     'monday_delivery_quantity': [10, 15, 5, 8, 12, 20],
#     'tuesday_delivery_quantity': [8, 10, 3, 6, 9, 15],
#     'wednesday_delivery_quantity': [12, 14, 7, 10, 15, 25],
#     'thursday_delivery_quantity': [15, 18, 8, 12, 18, 30],
#     'friday_delivery_quantity': [20, 22, 12, 15, 25, 40],
#     'saturday_delivery_quantity': [25, 30, 15, 20, 35, 50]
# }
#
# # 转换成 DataFrame
# df = pd.DataFrame(data)
#
# # 输出原始数据
# print("原始数据：")
# print(df)
#
# # 按照 mould_series 进行 product_code 聚合，并求和
# df_aggregated = df.groupby('mould_series').agg({
#     'monday_delivery_quantity': 'sum',
#     'tuesday_delivery_quantity': 'sum',
#     'wednesday_delivery_quantity': 'sum',
#     'thursday_delivery_quantity': 'sum',
#     'friday_delivery_quantity': 'sum',
#     'saturday_delivery_quantity': 'sum'
# }).reset_index()
#
# # 输出聚合结果
# print("\n按照 mould_series 聚合后的结果：")
# print(df_aggregated)
product_info_file = "../utils/产品型号交付限制.json"
with open(product_info_file, 'r', encoding='utf-8') as order_file:
    product_info = order_file.read()
input_dict = json_to_dataframes(product_info)
print(input_dict)
product_delivery_restriction = input_dict['product_delivery_restriction']
print(product_delivery_restriction)


mould_mapping = pd.merge(mould_mapping, product_delivery_restriction, on='mould_series', copy=False)


mould_mapping['order_quantity'] = mould_mapping['day_quantity'] * mould_mapping['unit_capacity']

# 计算 within_week_order_quantity
mould_mapping['within_week_order_quantity'] = mould_mapping[
    ['monday_delivery_quantity', 'tuesday_delivery_quantity',
     'wednesday_delivery_quantity', 'thursday_delivery_quantity',
     'friday_delivery_quantity', 'saturday_delivery_quantity']
].sum(axis=1)

# 更新 saturday_delivery_quantity
mask = mould_mapping['order_quantity'] > mould_mapping['within_week_order_quantity']
mould_mapping.loc[mask, 'saturday_delivery_quantity'] += mould_mapping.loc[mask, 'order_quantity'] - mould_mapping.loc[mask, 'within_week_order_quantity']

# 最终的结果
print(mould_mapping.to_string(index=False))

mould_mapping_melted = pd.melt(mould_mapping, id_vars=['mould_series', 'mould_cnt', 'unit_capacity', 'day_quantity', 'order_quantity', 'within_week_order_quantity'],
                               value_vars=['monday_delivery_quantity', 'tuesday_delivery_quantity', 'wednesday_delivery_quantity', 'thursday_delivery_quantity', 'friday_delivery_quantity', 'saturday_delivery_quantity'],
                               var_name='delivery_time', value_name='delivery_quantity')

mould_mapping_melted = mould_mapping_melted[mould_mapping_melted['delivery_quantity'] > 0]

# 将delivery_time中的周一到周六用0到5代替
mould_mapping_melted['delivery_time'] = mould_mapping_melted['delivery_time'].replace({
    'monday_delivery_quantity': 0,
    'tuesday_delivery_quantity': 1,
    'wednesday_delivery_quantity': 2,
    'thursday_delivery_quantity': 3,
    'friday_delivery_quantity': 4,
    'saturday_delivery_quantity': 5
})

print(mould_mapping_melted.to_string(index=False))
