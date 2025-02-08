import base64, requests
import time
import json

from requests.adapters import HTTPAdapter

from docs_api import version

session = requests.session()

session.mount("http://", HTTPAdapter(max_retries=2))
session.mount("https://", HTTPAdapter(max_retries=2))

'''
输入参数：
1. lp_file(str) LP模型文件名称。(示例："test.lp")，必填
2. access_token(str) 调用令牌，必填
3. task_type('Normal'|'MultiObj') 任务类型，必填，默认 'Normal'
5. task_name(str) 任务名称，必填（默认为'DefaultTask')
5. time_limit(int) 时间限制，非必填（默认300，单位/秒）
7. task_priority(int) 任务优先级 ，非必填（'1'/'2'/'3'，默认为'3'）
4. mpg(str) MIPGap，参数，必填
9. mip(str) Mip，非必填，默认为'0'
8. param_set(list) 多目标参数组，如果taskType入参为'MultiObj'，即必须提供一个多目标参数列表（例：parameterSet = ['MIPGap=0.001,TimeLimit=10', 'MIPGap=0.002,TimeLimit=20']），提供后可忽略其他Gurobi参数输入，例：mip，others
10. others(str) 其它参数，示例：‘Heuristics=0.07，MIPFocus=0’，非必填，默认为空
11. input_file 参数文件
返回值：任务ID
异常处理：如果发生异常，会raise Exception
'''


def submit_task(
        host: str,
        lp_file: str,
        access_token: str,
        task_type: str = 'Normal',
        task_name: str = 'DefaultTask',
        time_limit: int = 300,
        task_priority: int = 3,
        mpg: str = '',
        mip: str = '0',
        param_set: list = [],
        others: str = '',
        input_file: str = ''):
    with open(lp_file, 'r', encoding='utf-8') as lp_file_:
        response = session.post(host + "/v1.0/task/uploadFile", files={"file": lp_file_})
        file_id = process_response(response)
        print(file_id)

    input_file_id = '0'
    if input_file != '':
        with open(input_file, 'r', encoding='utf-8') as input_file_:
            response = session.post(host + "/v1.0/task/uploadFile", files={"file": input_file_})
            input_file_id = process_response(response)

    if task_type != 'Normal' and task_type != 'MultiObj':
        raise Exception('任务类型只支持Normal和MultiObj')

    param_set
    if task_type == 'MultiObj':
        if param_set.count() == 0:
            raise Exception('任务类型为MultiObj，param_set不能为空')

    if task_priority < 1 or task_priority > 3:
        raise Exception('任务优先级只支持1~3')

    task_json = {
        'name': task_name,
        'type': task_type,
        'priority': task_priority,
        'paramMip': mip,
        'paramGap': mpg,
        'paramTimeLimit': time_limit,
        'paramOthers': others,
        'paramSet': param_set,
        'modelFileId': file_id,
        'accessToken': access_token,
        'sdkVersion': version,
        'inputFileId': input_file_id
    }
    url = host + "/v1.0/task/submit"

    response = session.post(url, timeout=30, json=task_json)
    task_id = process_response(response)
    return task_id


'''
读取任务日志，转为字典形式
'''


def get_variables_dict(task_result, output_json_path):
    # 初始化两个字典映射
    m_vars = {}
    v_vars = {}

    # with open(file_path, mode='r', encoding='utf-8') as f:
    #     file_content = f.readlines()
    lines = task_result.splitlines()

    for line in lines:
        line = line.strip()
        if line.startswith('m') or line.startswith('v'):
            parts = line.split()
            var_name = parts[0]
            var_value = int(parts[1])

            if var_name.startswith('m'):
                m_vars[var_name] = var_value
            elif var_name.startswith('v'):
                v_vars[var_name] = var_value

    # 保存为 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump({'v_vars': v_vars, 'm_vars': m_vars}, json_file, ensure_ascii=False, indent=4)


'''
提交任务，输出任务日志
'''


def submit_task_and_wait(
        host: str,
        lp_file: str,
        access_token: str,
        task_type: str = 'Normal',
        task_name: str = 'DefaultTask',
        time_limit: int = 300,
        task_priority: int = 3,
        mpg: str = '',
        mip: str = '0',
        param_set: list = [],
        others: str = '',
        result_dir="./"):
    # 提交任务
    task_id = submit_task(host, lp_file, access_token, task_type, task_name, time_limit, task_priority, mpg,
                          mip, param_set, others)

    print(f'提交任务成功， 任务ID = {task_id}')

    print('-------开始打印日志--------')
    start = time.time()

    while True:

        # 读取任务信息
        task_detail = get_task_info(host, access_token, task_id)

        if task_detail.__contains__('log') and task_detail['log'] != '':
            print(task_detail['log'], end='')

        # 任务处理成功
        if task_detail['status'] == 'succeed':
            print('-------结束打印日志-------')
            print(f'任务执行完毕，耗时：{round(time.time() - start, 2)}秒，结果写入文件{result_dir}{task_id}.out')
            # with open(f'{result_dir}{task_id}.out', mode='w', encoding='utf-8') as f:
            #     f.write(task_detail['result'])
            print('求解任务结果', task_detail['result'])
            # file_path = f'{result_dir}{task_id}.out'
            output_json_path = f'{result_dir}{task_name}.json'
            print(task_name)
            get_variables_dict(task_detail['result'], output_json_path)
            break
        # 任务处理失败
        if task_detail['status'] == 'failed':
            print('-------结束打印日志-------')
            print(f'任务执行失败，耗时：{round(time.time() - start, 2)}秒')
            break
        if task_detail['status'] == 'canceled':
            print('-------结束打印日志-------')
            print(f'任务已被取消，耗时：{round(time.time() - start, 2)}秒')
            break

        if time.time() - start > time_limit:
            print(f'任务已运行{round(time.time() - start, 2)}秒，超过最大等待时间{time_limit}秒，将被取消')
            cancel(host, access_token, task_id)

        time.sleep(0.2)


'''

返回结构体
{
    "id": "" # 任务ID
    "name": "" # 任务名称
    "type": "" # 任务类型
    "status": "" # 任务状态

    "createAt": "" # 任务创建时间
    "startAt": "" # 开始时间
    "finishAt": "" # 完成时间

    "result": "" # 运行结果
    "log": "" # 输出的日志
    "error" : "" # 错误信息

}
'''


def get_task_info(host: str, access_token: str, task_id: str):
    url = host + "/v1.0/task/getTaskResult"

    response = session.get(url, timeout=30, params={
        'accessToken': access_token,
        'taskId': task_id,
    })

    http_result = process_response(response)
    return http_result


'''
返回值结构体
[{
    "id": "" # 任务ID
    "name": "" # 任务名称
    "type": "" # 任务类型
    "status": "" # 任务状态
    "priority": "" # 优先级

    "createAt": "" # 任务创建时间
}]
'''


def get_ongoing_tasks(host: str, access_token: str):
    url = host + "/v1.0/task/getOngoingTasks"
    response = session.get(url, params={
        'accessToken': access_token,
    })
    http_result = process_response(response)
    return http_result


'''
取消任务

1. host(str) 调用地址
2. access_token(str) 调用令牌，必填
3. task_id(str) 任务ID
'''


def cancel(host: str, access_token: str, task_id: str):
    url = host + "/v1.0/task/cancel"

    task_json = {
        'accessToken': access_token,
        'taskId': task_id,
    }

    response = session.post(url, timeout=30, json=task_json)

    http_result = process_response(response)

    return http_result


def process_response(response: requests.Response):
    if response.status_code != 200:
        raise Exception("调用远程接口错误，请稍后重试")

    json_data = response.json()
    if json_data['code'] != 200:
        raise Exception(json_data['message'])
    if 'data' in json_data:
        return json_data['data']
    return ''


# def encode(origional_str):
#     bytes_str = origional_str.encode('utf-8')
#     bytes_str = base64.b64encode(bytes_str)
#     serial_str = str(bytes_str, 'utf-8')
#     return serial_str
#
#
# def decode(serial_str):
#     bytes_str = base64.b64decode(serial_str)
#     original_str = bytes_str.decode(encoding='utf-8')
#     return original_str


if __name__ == "__main__":
    pass
