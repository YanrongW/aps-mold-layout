import pandas as pd
import json
import datetime
import os
from docs_api.api import submit_task_and_wait
from mold_layout.create_lp_model import (
    process_factory_data,
    calculate_online_mould_status,
    execute_linear_programming,
    get_gurobi_optimal_solution,
    transform_to_matrix,
    transform_production_results,
    split_and_process_results
)

class MouldArrangementSolver:
    def __init__(self, lp_file_path=None, gurobi_solver_path=None, file_name=None, mapping_file_name=None):
        self.lp_file_path = lp_file_path
        self.gurobi_solver_path = gurobi_solver_path
        self.file_name = file_name
        self.mapping_file_name = mapping_file_name

        ask_name = "mould_arrangement_lp"
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
        file_name = f"{ask_name}_{formatted_time}"

        self.file_name = file_name
        self.mapping_file_name = f"{self.file_name}_mode_code_mapping.json"

        # 初始化变量
        self.result_a = None
        self.result_b = None
        self.message = True

    def main_process(self, order_info, work_calendar_id=list(range(6))):
        # 1. 基本数据处理
        mould_mapping_melted, mould_online_state, max_fixture_count, \
        mould_to_bus, mould_info, mould_code_mapping = process_factory_data(order_info, self.mapping_file_name)

        # 检查 mould_info 是否为空数据框
        if mould_info.empty:
            self.message = False
            return

        mould_online_state_copy, mould_mapping_melted_copy = calculate_online_mould_status(mould_online_state,
                                                                                           mould_mapping_melted)

        # 2. 线性规划问题建模
        model = execute_linear_programming(work_calendar_id, mould_info, mould_to_bus, max_fixture_count,
                                           mould_mapping_melted_copy, mould_online_state_copy)


        self.lp_file_path = f"{self.file_name}.lp"
        model.write(self.lp_file_path)


    def gurobi_solve(self, host, access_token):
        if self.lp_file_path is not None:
            submit_task_and_wait(
                host=host,
                task_name=self.file_name,
                lp_file=self.lp_file_path,
                access_token=access_token,
                time_limit=5,
                task_type="Normal"
            )

            self.gurobi_solver_path = f"{self.file_name}.json"

            if os.path.exists(self.gurobi_solver_path):
                all_vars = get_gurobi_optimal_solution(self.gurobi_solver_path)
                v_vars, m_vars = all_vars['v_vars'], all_vars['m_vars']

                production_result_df = transform_to_matrix(v_vars, self.mapping_file_name)
                mould_result_df = transform_to_matrix(m_vars, self.mapping_file_name)

                production_result_a, mould_result_a = split_and_process_results(production_result_df, mould_result_df, 'A')
                production_result_b, mould_result_b = split_and_process_results(production_result_df, mould_result_df, 'B')

                result_a = transform_production_results(production_result_a, mould_result_a, 'A')
                result_b = transform_production_results(production_result_b, mould_result_b, 'B')

                self.result_a = result_a
                self.result_b = result_b
            else:
                self.result_a = None
                self.result_b = None

    def solve(self, order_info, work_calendar_id, host, access_token):
        self.main_process(order_info, work_calendar_id)
        if self.message is False:
            return
        self.gurobi_solve(host, access_token)

    def delete_intermediate_files(self):
        # 删除 lp 文件
        if self.lp_file_path and os.path.exists(self.lp_file_path):
            os.remove(self.lp_file_path)
        # 删除 gurobi solver 文件
        if self.gurobi_solver_path and os.path.exists(self.gurobi_solver_path):
            os.remove(self.gurobi_solver_path)
            # 删除 模具系列与编码 map 文件
        if self.mapping_file_name and os.path.exists(self.mapping_file_name):
            os.remove(self.mapping_file_name)


if __name__ == "__main__":
    order_info_file = "../utils/模具配置.json"
    with open(order_info_file, 'r', encoding='utf-8') as order_file:
        order_info = json.loads(order_file.read())

    work_calendar_id = [1, 2, 3, 4, 5, 6]
    host = "http://docs-service.qd-aliyun-dmz-ack-internal.haier.net/"
    access_token = "8bd0f0a3-1139-4518-93f5-79dcd43c3b16"


    solver = MouldArrangementSolver()
    solver.solve(order_info, work_calendar_id, host, access_token)
    solver.delete_intermediate_files()

    result_a = solver.result_a
    result_b = solver.result_b

    print(result_a)
    print(result_b)

    if result_b is not None:
        df_quantitative = pd.DataFrame.from_dict(result_b['B_quantitative_results'])
        df_mould_arrangement = pd.DataFrame.from_dict(result_b['B_mould_layout_results'])

        print("B线定量结果:")
        print(df_quantitative)

        print("\nB线模具排布结果:")
        print(df_mould_arrangement)
