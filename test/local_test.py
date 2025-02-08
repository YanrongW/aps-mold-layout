import requests
import json
import os

# 获取当前工作目录
current_directory = os.getcwd()
# 获取上级目录
parent_directory = os.path.abspath(os.path.join(current_directory, ".."))


# 构建相对路径
order_info_file = "utils/模具配置.json"

# 使用 os.path.join 构建完整路径
order_info_path = os.path.join(parent_directory, order_info_file)

with open(order_info_path, 'r', encoding='utf-8') as order_file:
    order_info = json.load(order_file)

url = "http://localhost:5000/solve-mould-arrangement"  # Replace with the actual server IP and port

data = {
    "order_info": order_info,
    "work_calendar_id": list(range(6)),
    "host": "http://docs-service.qd-aliyun-dmz-ack-internal.haier.net/",
    "access_token": "8bd0f0a3-1139-4518-93f5-79dcd43c3b16"
}

response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Success!")
    # print("Result A:", result['data']['result_a'])
    print("Result B:", result['data']['result_b']['B_mould_layout_results'])
else:
    print("Error:", response.status_code, response.text)
