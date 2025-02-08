import pandas as pd
import json
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from collections import defaultdict
import warnings
import tempfile
# from prettytable import PrettyTable
from docs_api.api import submit_task_and_wait

warnings.filterwarnings('ignore')


def json_to_dataframes(json_data):
    # 遍历每个Sheet表单的JSON数据并构建数据框
    dataframes = {}
    for sheet_name in json_data.keys():
        # 从JSON数据创建数据框
        df = pd.DataFrame(json_data[sheet_name])
        # 将数据框存储到字典中
        dataframes[sheet_name] = df

    return dataframes


def process_factory_data(order_info, mapping_file_name):
    # 处理输入订单数据
    input_dict = json_to_dataframes(order_info)

    # 待生产订单
    all_order_df = input_dict['mould_configs']
    if input_dict['mould_configs'].empty:
        all_order_df = pd.DataFrame(columns=['mould_series', 'mould_cnt', 'unit_capacity', 'day_quantity', 'bus_code'])
    else:
        all_order_df.columns = ['mould_series', 'mould_cnt', 'unit_capacity', 'day_quantity', 'bus_code']
    all_order_df['order_quantity'] = all_order_df['day_quantity'] * all_order_df['unit_capacity']
    all_order_df['total_capacity'] = all_order_df['mould_cnt'] * all_order_df['unit_capacity']

    # 建立字典映射
    row_mark = pd.unique(all_order_df['mould_series'])
    mould_code_mapping = {code: 'm' + str(i) for i, code in enumerate(row_mark)}
    all_order_df['mould_code'] = all_order_df['mould_series'].map(mould_code_mapping)
    with open(mapping_file_name, 'w') as json_file:
        json.dump(mould_code_mapping, json_file, indent=4)

    # 模具与线体的约束关系
    mould_to_bus = {}
    for _, row in all_order_df[['mould_code', 'bus_code']].iterrows():
        mould_to_bus[row['mould_code']] = row['bus_code']

    # 产品型号交付限制
    product_delivery_restriction = input_dict['product_delivery_restriction']

    if input_dict['product_delivery_restriction'].empty:
        product_delivery_restriction = pd.DataFrame(columns=['mould_series', 'product_code', 'sunday_delivery_quantity',
                                                             'monday_delivery_quantity', 'tuesday_delivery_quantity',
                                                             'wednesday_delivery_quantity', 'thursday_delivery_quantity',
                                                             'friday_delivery_quantity', 'saturday_delivery_quantity'])
    else:
        product_delivery_restriction.columns = ['mould_series', 'product_code', 'sunday_delivery_quantity', 'monday_delivery_quantity',
                                                'tuesday_delivery_quantity', 'wednesday_delivery_quantity', 'thursday_delivery_quantity',
                                                'friday_delivery_quantity', 'saturday_delivery_quantity']
    mould_delivery_restriction = product_delivery_restriction.groupby('mould_series').agg({
        'sunday_delivery_quantity': 'sum',
        'monday_delivery_quantity': 'sum',
        'tuesday_delivery_quantity': 'sum',
        'wednesday_delivery_quantity': 'sum',
        'thursday_delivery_quantity': 'sum',
        'friday_delivery_quantity': 'sum',
        'saturday_delivery_quantity': 'sum'
    }).reset_index()

    mould_mapping = pd.merge(all_order_df, mould_delivery_restriction, on='mould_series', how='left', copy=False)
    # 处理缺失值
    mould_mapping.fillna(0, inplace=True)

    # 计算周内订单量
    mould_mapping['within_week_order_quantity'] = mould_mapping[['sunday_delivery_quantity', 'monday_delivery_quantity',
                                                                 'tuesday_delivery_quantity','wednesday_delivery_quantity',
                                                                 'thursday_delivery_quantity', 'friday_delivery_quantity',
                                                                 'saturday_delivery_quantity']].sum(axis=1)

    # 更新 saturday_delivery_quantity
    mask = mould_mapping['order_quantity'] > mould_mapping['within_week_order_quantity']
    mould_mapping.loc[mask, 'saturday_delivery_quantity'] += mould_mapping.loc[mask, 'order_quantity'] - \
                                                             mould_mapping.loc[mask, 'within_week_order_quantity']
    # 最终的结果
    print(mould_mapping.to_string(index=False))

    mould_mapping_melted = pd.melt(mould_mapping, id_vars=['mould_series', 'mould_code', 'mould_cnt',
                                                           'unit_capacity', 'day_quantity', 'order_quantity',
                                                           'within_week_order_quantity', 'total_capacity', 'bus_code'],
                                   value_vars=['sunday_delivery_quantity', 'monday_delivery_quantity', 'tuesday_delivery_quantity',
                                               'wednesday_delivery_quantity', 'thursday_delivery_quantity',
                                               'friday_delivery_quantity', 'saturday_delivery_quantity'],
                                   var_name='delivery_time', value_name='delivery_quantity')

    mould_mapping_melted = mould_mapping_melted[mould_mapping_melted['delivery_quantity'] > 0]

    # 将delivery_time中的周日到周六用0到6代替
    mould_mapping_melted['delivery_time'] = mould_mapping_melted['delivery_time'].replace({
        'sun_delivery_quantity': 0,
        'monday_delivery_quantity': 1,
        'tuesday_delivery_quantity': 2,
        'wednesday_delivery_quantity': 3,
        'thursday_delivery_quantity': 4,
        'friday_delivery_quantity': 5,
        'saturday_delivery_quantity': 6
    })

    mould_info = mould_mapping_melted[['mould_series', 'mould_code', 'mould_cnt',
                                       'unit_capacity', 'total_capacity', 'bus_code']].drop_duplicates()

    # 上周在线模具状态
    if input_dict['mould_online_state_last_week'].empty:
        mould_online_state = pd.DataFrame(columns=['mould_series', 'online_mould_cnt',
                                                   'unit_capacity', 'online_total_capacity', 'mould_code'])
    else:
        mould_online_state = input_dict['mould_online_state_last_week']
        mould_online_state.columns = ['mould_series', 'online_mould_cnt', 'unit_capacity']
        mould_online_state['online_total_capacity'] = mould_online_state.online_mould_cnt * mould_online_state.unit_capacity  # 计算出初始模具的产能
        mould_online_state = pd.merge(mould_online_state, mould_info[['mould_series', 'mould_code']],
                                      how='left', on='mould_series', copy=False)

    # 中小单的最大夹具数量
    max_fixture_count = input_dict['fixture_count'].loc[0, 'max_fixture_count']

    return mould_mapping_melted, mould_online_state, max_fixture_count, mould_to_bus, mould_info, mould_code_mapping


def calculate_online_mould_status(mould_online_state, mould_mapping_melted):
    mould_online_state_copy = mould_online_state.copy()
    mould_mapping_melted_copy = mould_mapping_melted.copy()

    return mould_online_state_copy, mould_mapping_melted_copy


# step1. 变量设计
def create_lp_problem(mould_info, work_calendar_id):
    # 获取线体编码列表
    bus_list = ['A', 'B']

    # 定义模型问题
    model = gp.Model("LP Problem")

    # 生成规划变量
    row_mark = pd.unique(mould_info.mould_code)  # 行变量设定，即模具编码
    col_mark0 = []  # 列变量设定，即线体编码_时间编码，比如A_0、B_0
    col_mark = []
    for time_idx in work_calendar_id:
        for bus_idx in bus_list:
            col_mark0.append(str(bus_idx) + '_' + str(time_idx))
    col_mark = col_mark0 + ['delay']

    choices = {(m, c): model.addVar(vtype=GRB.INTEGER, name=f'v_{m}_{c}', lb=0) for m in row_mark for c in col_mark}
    v_variable = pd.DataFrame(index=row_mark, columns=col_mark, data=0)

    for (m, c), var in choices.items():
        v_variable.at[m, c] = var

    return model, v_variable, col_mark0, col_mark, row_mark


# step2. 单变量约束
def apply_univariate_constraints(model, v_variable, col_mark0, row_mark, mould_to_bus, mould_info, work_calendar_id):
    # 2.1 单变量约束_模具与线体的绑定关系_确保模具只能在与其绑定的线体上进行生产
    for bus_and_time in col_mark0:  # 遍历列标签，即线体_时间组合，形如A_0
        for mould_code, single_var in v_variable[bus_and_time].items():  # 遍历模具及变量
            if bus_and_time.split('_')[0] not in mould_to_bus[mould_code]:
                model.addConstr(single_var == 0)

    # 2.2 单变量约束_同一天的少于发泡上限_确保同一天该模具的总生产量不超过其发泡上限
    for mould_code in row_mark:  # 遍历模具
        for calendar_id in work_calendar_id:  # 遍历发货时间
            bus_and_time_list = []  # 同一天的时间标签列表容器
            for bus_and_time in col_mark0:  # 遍历列标签
                if str(calendar_id) in bus_and_time:
                    bus_and_time_list.append(bus_and_time)  # 时间容器填充
            # 限制条件添加
            model.addConstr(gp.quicksum(v_variable.loc[mould_code, bus_and_time_list]) <= mould_info[
                mould_info.mould_code == mould_code].total_capacity.sum())
    return model, v_variable


# step3. 行约束
def add_row_constraints(model, v_variable, mould_mapping_melted_copy, col_mark0):
    # 3.1 行约束_总产量等于总需求
    for mould_code, bus_and_time_info in v_variable.iterrows():
        tmp_sum = mould_mapping_melted_copy[mould_mapping_melted_copy.mould_code == mould_code].order_quantity.sum()
        model.addConstr(gp.quicksum(bus_and_time_info) == tmp_sum)

    # 3.2 行约束_发货时间约束
    for mould_code, bus_and_time_info in v_variable.iterrows():
        # 获取当前模具的任务
        cur_demand_df = mould_mapping_melted_copy[mould_mapping_melted_copy.mould_code == mould_code]
        cur_max_num = 0
        # 遍历当前模具的要求生产的日期清单
        for _, cur_demand in cur_demand_df.iterrows():
            # 获取在此日期之前的所有index
            index_list = [i for i in col_mark0 if int(i.split('_')[1]) <= cur_demand['delivery_time']]
            cur_max_num += cur_demand['order_quantity']
            # 确保模具的延单量和在约束时间范围内的产量之和>=需求量之和
            model.addConstr(bus_and_time_info.delay + gp.quicksum(
                bus_and_time_info[bus_and_time_info.index.isin(index_list)]) >= cur_max_num)

    return model, v_variable


# step4. 列约束
def add_column_constraints(model, v_variable, col_mark0, mould_info, max_fixture_count):
    # 4.1 列约束_线体日产能_确保该线体在该时间节点的总产量不超过预定的日产能
    # for bus_and_time in col_mark0:
    #     bus_code = bus_and_time.split('_')[0]
    #     model.addConstr(gp.quicksum(v_variable[bus_and_time]) <= bus_dict[bus_code][0])

    # 4.2 列约束_线体夹具数_确保该线体在该时间节点的模具套数不超过预定的夹具数
    for bus_and_time in col_mark0:
        bus_code = bus_and_time.split('_')[0]
        used_fixture_cnt = 0
        for mould_code, single_val in v_variable[bus_and_time].items():
            used_fixture_cnt += single_val * (1 / mould_info[mould_info.mould_code == mould_code].unit_capacity.sum())
        model.addConstr(used_fixture_cnt <= max_fixture_count)

    return model, v_variable


# step5. 目标函数
def add_objective_functions(model, v_variable, col_mark0, row_mark, mould_mapping_melted_copy, mould_info,
                            mould_online_state_copy, max_fixture_count, work_calendar_id):
    # 5.1 最小化总和成本_不延期，延期成本最高
    min_cost = gp.quicksum(v_variable.delay)

    # 5.2 最小化总和成本_不拖拉，尽量集中生产_给定上限，尽可能少使用夹具
    m_int = {(mould_code, bus_and_time): model.addVar(vtype=GRB.INTEGER, name=f'm_{mould_code}_{bus_and_time}', lb=0)
             for mould_code in row_mark for bus_and_time in col_mark0}
    m_variable = pd.DataFrame(index=row_mark, columns=col_mark0, data=0)
    for (m, c), var in m_int.items():
        m_variable.at[m, c] = var

    f_int = {(mould_code, bus_and_time): model.addVar(vtype=GRB.INTEGER, name=f'f_{mould_code}_{bus_and_time}', lb=0)
             for mould_code in row_mark for bus_and_time in col_mark0}
    f_variable = pd.DataFrame(index=row_mark, columns=col_mark0, data=0)
    for (m, c), var in f_int.items():
        f_variable.at[m, c] = var

    # 每个模具在每天使用夹具数 不小于 当天该模具生产量/模具单位产能
    for mould_code, bus_and_time_info in m_variable.iterrows():
        day_quantity = 0
        for bus_and_time, m_var in bus_and_time_info.items():
            model.addConstr(m_var >= v_variable.loc[mould_code, bus_and_time] * (
                    1 / mould_info[mould_info.mould_code == mould_code].unit_capacity.sum()))
            day_quantity += m_var
        model.addConstr(day_quantity ==
                        mould_mapping_melted_copy[mould_mapping_melted_copy.mould_code == mould_code].day_quantity.sum())

    min_model = gp.quicksum(m_variable.loc[mould_code, bus_and_time]
                            for mould_code in row_mark for bus_and_time in col_mark0)

    # 给定上限，尽可能少使用夹具
    min_fixture = 0
    for bus_and_time in col_mark0:
        cur_day_index = int(bus_and_time.split('_')[1])
        if cur_day_index == 0:
            continue
        else:
            for prev_day_index in range(cur_day_index - 1, -1, -1):
                bus_and_prev_day = bus_and_time.split('_')[0] + '_' + str(prev_day_index)
                if bus_and_prev_day not in col_mark0:
                    continue
                model.addConstr(gp.quicksum(f_variable[bus_and_time])
                                >= gp.quicksum(m_variable[bus_and_time]) - gp.quicksum(m_variable[bus_and_prev_day]))
                model.addConstr(gp.quicksum(f_variable[bus_and_time])
                                >= gp.quicksum(m_variable[bus_and_prev_day]) - gp.quicksum(m_variable[bus_and_time]))
                min_fixture += gp.quicksum(f_variable[bus_and_time])

    # 细粒度取整操作，不超过每天的夹具数上限
    for bus_and_time in col_mark0:
        bus_code = bus_and_time.split('_')[0]
        model.addConstr(gp.quicksum(m_variable[bus_and_time]) <= max_fixture_count)

    # 5.3 最小化日模具变化_相邻两天的模具使用情况变化越少越好
    c_int = {(mould_code, bus_and_time): model.addVar(vtype=GRB.INTEGER, name=f'c_{mould_code}_{bus_and_time}', lb=0)
             for mould_code in row_mark for bus_and_time in col_mark0}
    c_variable = pd.DataFrame(index=row_mark, columns=col_mark0, data=0)
    for (m, c), var in c_int.items():
        c_variable.at[m, c] = var

    result_dict = dict()
    for item in col_mark0:
        bus_code, day = item.split('_')
        if bus_code not in result_dict:
            result_dict[bus_code] = []
            result_dict[bus_code].append(int(day))
        else:
            result_dict[bus_code].append(int(day))
    max_values = {key: max(values) for key, values in result_dict.items()}

    min_exchange = 0
    for mould_code, bus_and_time_info in c_variable.iterrows():
        for bus_and_time, c_var in bus_and_time_info.items():
            # 生产第一天使用初始状态的模具数量
            if int(bus_and_time.split('_')[1]) == min(work_calendar_id):
                m_var = mould_online_state_copy.loc[(mould_online_state_copy.mould_code == mould_code), 'online_mould_cnt'].sum()
            else:
                previous_index = work_calendar_id.index(int(bus_and_time.split('_')[1])) - 1
                bus_and_previous_day = bus_and_time.split('_')[0] + '_' + str(work_calendar_id[previous_index])
                m_var = m_variable.loc[mould_code, bus_and_previous_day]
            if int(bus_and_time.split('_')[1]) == max_values[bus_and_time.split('_')[0]]:
                model.addConstr(c_var >= m_variable.loc[mould_code, bus_and_time] - m_var)
                model.addConstr(c_var >= m_var - m_variable.loc[mould_code, bus_and_time])
                model.addConstr(c_var >= - m_variable.loc[mould_code, bus_and_time])
                model.addConstr(c_var >= m_variable.loc[mould_code, bus_and_time])
            else:
                model.addConstr(c_var >= m_variable.loc[mould_code, bus_and_time] - m_var)
                model.addConstr(c_var >= m_var - m_variable.loc[mould_code, bus_and_time])
            min_exchange += c_var

    # 添加优化目标函数
    model.setObjective(min_exchange * 180 + min_cost * 1000 + min_model * 60 + min_fixture * 10, sense=GRB.MINIMIZE)

    return model


def execute_linear_programming(work_calendar_id, mould_info, mould_to_bus, max_fixture_count,
                               mould_mapping_melted_copy, mould_online_state_copy):
    # step1. 变量设计
    model, v_variable, col_mark0, col_mark, row_mark = create_lp_problem(mould_info, work_calendar_id)
    # step2. 单变量约束
    # 2.1 单变量约束_模具与线体的绑定关系_确保模具只能在与其绑定的线体上进行生产
    # 2.2 单变量约束_同一天的少于发泡上限_确保同一天该模具的总生产量不超过其发泡上限
    model, v_variable = apply_univariate_constraints(model, v_variable, col_mark0, row_mark, mould_to_bus,
                                                     mould_info, work_calendar_id)

    # step3. 行约束
    # 3.1 行约束_总产量等于总需求
    # 3.2 行约束_发货时间约束
    model, v_variable = add_row_constraints(model, v_variable, mould_mapping_melted_copy, col_mark0)
    # step4. 列约束
    # 4.1 列约束_线体日产能_确保该线体在该时间节点的总产量不超过预定的日产能
    # 4.2 列约束_线体夹具数_确保该线体在该时间节点的模具套数不超过预定的夹具数
    model, v_variable = add_column_constraints(model, v_variable, col_mark0, mould_info, max_fixture_count)
    # step5. 目标函数
    # 5.1 最小化总和成本_不延期，延期成本最高
    # 5.2 最小化总和成本_不拖拉，尽量集中生产
    # 5.3 最小化日模具变化_相邻两天的模具使用情况变化越少越好
    model = add_objective_functions(model, v_variable, col_mark0, row_mark, mould_mapping_melted_copy, mould_info, mould_online_state_copy,
                                    max_fixture_count, work_calendar_id)


    return model


def model_process(order_info, mapping_file_name, work_calendar_id=list(range(6))):
    # 1. 基本数据处理
    mould_mapping_melted, mould_online_state, max_fixture_count, \
    mould_to_bus, mould_info, mould_code_mapping = process_factory_data(order_info, mapping_file_name)
    mould_online_state_copy, mould_mapping_melted_copy = calculate_online_mould_status(mould_online_state, mould_mapping_melted)

    # 2. 线性规划问题建模
    # 到这一步得到lp文件后，后面的内容都可以不要，要调用服务使用gurobi求解器求解问题
    model = execute_linear_programming(work_calendar_id, mould_info, mould_to_bus, max_fixture_count,
                                       mould_mapping_melted_copy, mould_online_state_copy)
    model.write('is_not_main_mould_arrangement.lp')

    pass


def get_gurobi_optimal_solution(json_file_name):
    with open(json_file_name, 'r') as json_file:
        vars_mapping = json.load(json_file)
    return vars_mapping


def transform_to_matrix(vars, mapping_file_name):
    mould_time_info = defaultdict(dict)

    for key, value in vars.items():
        if key.split('_')[-1] != 'delay':
            _, mould_code, bus, time = key.split('_')
            bus_and_time = '_'.join([bus, time])
            mould_time_info[mould_code][bus_and_time] = value
        else:
            _, mould_code, delay = key.split('_')
            mould_time_info[mould_code][delay] = value

    for mould_code in mould_time_info:
        if 'delay' not in mould_time_info[mould_code]:
            mould_time_info[mould_code]['delay'] = 0

    with open(mapping_file_name, 'r') as json_file:
        loaded_mould_code_mapping = json.load(json_file)
    mould_code_mapping = {v: k for k, v in loaded_mould_code_mapping.items()}

    df = pd.DataFrame(mould_time_info).T
    df = df.astype(int)
    df.index = df.index.map(lambda x: mould_code_mapping.get(x, x))

    return df


def transform_to_original_table(original_df):
    original_df.index = original_df.index.astype(str)

    result_df = []
    for i in range(len(original_df)):
        row_data = {}
        current_index = original_df.index[i]
        for column in original_df.columns:
            row_data[column] = [current_index] * original_df[column][i]
        max_len = max(len(value) for value in row_data.values())
        # 将长度不足的值填充为None
        row_data = {key: value + [''] * (max_len - len(value)) for key, value in row_data.items()}
        row_df = pd.DataFrame(row_data)
        if i == 0:
            result_df = row_df
        else:
            result_df = pd.concat([result_df, row_df], ignore_index=True)

    return result_df


def transform_to_online_table(original_df):
    original_df.replace('', np.nan, inplace=True)
    new_df = pd.DataFrame(columns=original_df.columns)
    all_result = []
    for index, col in enumerate(original_df.columns):
        for i in range(len(original_df[col])):
            mould_code = original_df[col][i]
            prev_mould_code = None
            if pd.notna(mould_code):
                prev_mould_code = original_df.iloc[i, index - 1] if index > 0 else None
                if mould_code != prev_mould_code:
                    row_data = [''] * len(original_df.columns)
                    row_data[index] = mould_code
                    length = 1

                    for j in range(index + 1, len(original_df.columns)):
                        if original_df.iloc[i, j] == mould_code:
                            row_data[j] = mould_code
                            length += 1
                        else:
                            break
                    start_index = index
                    # length = len([item for item in row_data if item != ''])
                    # print(row_data, start_index, length)
                    res = [row_data, start_index, length]
                    all_result.append(res)

    for i in range(len(all_result)):
        row_data = all_result[i][0]
        start_index = all_result[i][1]
        length = all_result[i][2]
        if new_df.empty:
            new_df = new_df._append(pd.Series(row_data, index=new_df.columns), ignore_index=True)
        else:
            placed = False
            for new_index, row in new_df.iterrows():
                elements = new_df.iloc[new_index, start_index:start_index + length]
                if all(value == '' for value in elements):
                    new_df.iloc[new_index, start_index:start_index + length] = row_data[start_index:start_index + length]
                    placed = True
                    break
            if not placed:
                new_df = new_df._append(pd.Series(row_data, index=new_df.columns), ignore_index=True)
    return new_df


# def print_pretty_table_with_index(df):
#     pretty_table = PrettyTable()
#     pretty_table.field_names = ['Index'] + list(df.columns)
#
#     for index, row in df.iterrows():
#         pretty_table.add_row([index] + list(row))
#
#     return pretty_table


def split_and_process_results(production_result_df, mould_result_df, label):
    # 拆分定量结果
    production_result = production_result_df.filter(regex=f'^{label}_')
    production_result['delay'] = production_result_df['delay']  # 添加delay列
    production_result = production_result[~(production_result.iloc[:, :-1] == 0).all(axis=1)]

    # 拆分模具排布结果
    mould_result = mould_result_df.filter(regex=f'^{label}_')  # 筛选出以A开头的列
    mould_result['delay'] = mould_result_df['delay']  # 添加delay列
    mould_result = mould_result[~(mould_result.iloc[:, :-1] == 0).all(axis=1)]

    return production_result, mould_result


def transform_production_results(production_result, mould_result, line_label):
    result_dict = {}
    if not production_result.empty:
        result_dict[f"{line_label}_quantitative_results"] = production_result.to_dict()

        if not mould_result.empty:
            mould_result_transformed = transform_to_original_table(mould_result)
            mould_online_result = transform_to_online_table(mould_result_transformed)
            result_dict[f"{line_label}_mould_layout_results"] = mould_online_result.to_dict()
        else:
            result_dict[f"{line_label}_mould_layout_results"] = None
    else:
        result_dict[f"{line_label}_quantitative_results"] = None
        result_dict[f"{line_label}_mould_layout_results"] = None

    return result_dict


# def print_production_results(production_result, mould_result, line_label):
#     if not production_result.empty:
#         print(f"{line_label}线定量结果:")
#         print(print_pretty_table_with_index(production_result))
#
#         if not mould_result.empty:
#             print(f"{line_label}线原始模具排布结果:")
#             print(print_pretty_table_with_index(mould_result))
#
#             mould_result_transformed = transform_to_original_table(mould_result)
#             mould_online_result = transform_to_online_table(mould_result_transformed)
#
#             print(f"{line_label}线最终模具排布结果:")
#             print(print_pretty_table_with_index(mould_online_result))
#         else:
#             print(f"{line_label}线模具排布结果:")
#             print(print_pretty_table_with_index(mould_result))
#     else:
#         print(f"{line_label}线定量结果及模具排布结果为空")


def gurobi_solve(host, access_token, file_path):
    submit_task_and_wait(
        host=host,
        task_name="mould-arrangement-test",
        lp_file=file_path,
        access_token=access_token,
        time_limit=5,
        task_type="Normal"
    )

    json_file_name = "gurobi_optimal_vars.json"
    all_vars = get_gurobi_optimal_solution(json_file_name)
    v_vars, m_vars = all_vars['v_vars'], all_vars['m_vars']

    production_result_df = transform_to_matrix(v_vars)
    mould_result_df = transform_to_matrix(m_vars)

    production_result_a, mould_result_a = split_and_process_results(production_result_df, mould_result_df, 'A')
    production_result_b, mould_result_b = split_and_process_results(production_result_df, mould_result_df, 'B')

    result_a = transform_production_results(production_result_a, mould_result_a, 'A')
    result_b = transform_production_results(production_result_b, mould_result_b, 'B')

    return result_a, result_b
