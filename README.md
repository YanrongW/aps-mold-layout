# asp-mold-layout

中小单模具排布。

## 目录

- [安装](#安装)
- [使用](#使用)
- [API 接口](#api-接口)
- [贡献](#贡献)
- [许可证](#许可证)

## 安装

依赖项的说明。

```bash
pip install -r requirements.txt
```

## 使用

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app -preload
```

详细参考test文件夹下local_test.py文件
```python
import requests

url = "http://localhost:8000/solve-mould-arrangement"  # Replace with the actual server IP and port
order_info = '工厂订单信息json字符串'

data = {
    "order_info": order_info,
    "work_calendar_id": list(range(6)),
    "host": "http://docs-service.qd-aliyun-dmz-ack-internal.haier.net/",
    "access_token": "8bd0f0a3-1139-4518-93f5-79dcd43c3b16"
}

response = requests.post(url, json=data)
```

## API 接口

如果适用，提供有关 API 接口的详细信息以及如何进行请求的说明。

```http
POST /solve-mould-arrangement
```

示例请求负载：

```json
{
    "order_info": "订单信息json",
    "work_calendar_id": [0, 1, 2, 3, 4, 5],
    "host": "http://docs-service.qd-aliyun-dmz-ack-internal.haier.net/",
    "access_token": "8bd0f0a3-1139-4518-93f5-79dcd43c3b16"
}
```

示例响应：

```json
{
    "result_a": {
        "A\u7ebf\u5b9a\u91cf\u7ed3\u679c": null,
        "A\u7ebf\u6a21\u5177\u6392\u5e03\u7ed3\u679c": null
    },
    "result_b": {
        "B\u7ebf\u5b9a\u91cf\u7ed3\u679c": {
            "B_0": {
                "BCD4501": 0,
                "BCD465": 80,
                "BCD500": 0,
                "5618\u4e24\u95e8": 100,
                "BCD530": 100,
                "BCD580": 80,
                "5620\u4e09\u95e8": 24,
                "5620\u4e24\u95e8": 24,
                "BCD320": 87
            },
            "B_1": {
                "BCD4501": 0,
                "BCD465": 100,
                "BCD500": 0,
                "5618\u4e24\u95e8": 100,
                "BCD530": 74,
                "BCD580": 80,
                "5620\u4e09\u95e8": 100,
                "5620\u4e24\u95e8": 100,
                "BCD320": 100
            },
            "B_2": {
                "BCD4501": 120,
                "BCD465": 100,
                "BCD500": 0,
                "5618\u4e24\u95e8": 100,
                "BCD530": 100,
                "BCD580": 123,
                "5620\u4e09\u95e8": 100,
                "5620\u4e24\u95e8": 100,
                "BCD320": 100
            },
            "B_3": {
                "BCD4501": 84,
                "BCD465": 100,
                "BCD500": 120,
                "5618\u4e24\u95e8": 100,
                "BCD530": 0,
                "BCD580": 160,
                "5620\u4e09\u95e8": 100,
                "5620\u4e24\u95e8": 100,
                "BCD320": 100
            },
            "B_4": {
                "BCD4501": 120,
                "BCD465": 100,
                "BCD500": 120,
                "5618\u4e24\u95e8": 100,
                "BCD530": 0,
                "BCD580": 160,
                "5620\u4e09\u95e8": 0,
                "5620\u4e24\u95e8": 0,
                "BCD320": 100
            },
            "B_5": {
                "BCD4501": 120,
                "BCD465": 0,
                "BCD500": 120,
                "5618\u4e24\u95e8": 194,
                "BCD530": 0,
                "BCD580": 80,
                "5620\u4e09\u95e8": 0,
                "5620\u4e24\u95e8": 0,
                "BCD320": 0
            },
            "delay": {
                "BCD4501": 0,
                "BCD465": 0,
                "BCD500": 0,
                "5618\u4e24\u95e8": 0,
                "BCD530": 0,
                "BCD580": 0,
                "5620\u4e09\u95e8": 0,
                "5620\u4e24\u95e8": 0,
                "BCD320": 0
            }
        },
        "B\u7ebf\u6a21\u5177\u6392\u5e03\u7ed3\u679c": {
            "B_0": {
                "0": "BCD465",
                "1": "5618\u4e24\u95e8",
                "2": "BCD530",
                "3": "BCD580",
                "4": "5620\u4e09\u95e8",
                "5": "5620\u4e24\u95e8",
                "6": "BCD320",
                "7": "",
                "8": ""
            },
            "B_1": {
                "0": "BCD465",
                "1": "5618\u4e24\u95e8",
                "2": "BCD530",
                "3": "BCD580",
                "4": "5620\u4e09\u95e8",
                "5": "5620\u4e24\u95e8",
                "6": "BCD320",
                "7": "",
                "8": ""
            },
            "B_2": {
                "0": "BCD465",
                "1": "5618\u4e24\u95e8",
                "2": "BCD530",
                "3": "BCD580",
                "4": "5620\u4e09\u95e8",
                "5": "5620\u4e24\u95e8",
                "6": "BCD320",
                "7": "BCD4501",
                "8": "BCD580"
            },
            "B_3": {
                "0": "BCD465",
                "1": "5618\u4e24\u95e8",
                "2": "BCD500",
                "3": "BCD580",
                "4": "5620\u4e09\u95e8",
                "5": "5620\u4e24\u95e8",
                "6": "BCD320",
                "7": "BCD4501",
                "8": "BCD580"
            },
            "B_4": {
                "0": "BCD465",
                "1": "5618\u4e24\u95e8",
                "2": "BCD500",
                "3": "BCD580",
                "4": "",
                "5": "",
                "6": "BCD320",
                "7": "BCD4501",
                "8": "BCD580"
            },
            "B_5": {
                "0": "5618\u4e24\u95e8",
                "1": "5618\u4e24\u95e8",
                "2": "BCD500",
                "3": "BCD580",
                "4": "",
                "5": "",
                "6": "",
                "7": "BCD4501",
                "8": ""
            },
            "delay": {
                "0": "",
                "1": "",
                "2": "",
                "3": "",
                "4": "",
                "5": "",
                "6": "",
                "7": "",
                "8": ""
            }
        }
    }
}
```

## 排布结果展示

要将响应结果转换为如下形式：

B线定量结果:

|        | B_0 | B_1 | B_2 | B_3 | B_4 | B_5 | delay |
|--------|-----|-----|-----|-----|-----|-----|-------|
| BCD4501|  0  |  0  | 120 |  84 | 120 | 120 |   0   |
| BCD465 | 80  | 100 | 100 | 100 | 100 |  0  |   0   |
| BCD500 |  0  |  0  |  0  | 120 | 120 | 120 |   0   |
|5618两门| 100 | 100 | 100 | 100 | 100 | 194 |   0   |
| BCD530 | 100 |  74 | 100 |  0  |  0  |  0  |   0   |
| BCD580 |  80 |  80 | 123 | 160 | 160 |  80 |   0   |
|5620三门 |  24 | 100 | 100 | 100 |  0  |  0  |   0   |
|5620两门 |  24 | 100 | 100 | 100 |  0  |  0  |   0   |
| BCD320 |  87 | 100 | 100 | 100 | 100 |  0  |   0   |

B线模具排布结果:

|      | B_0   | B_1   | B_2   | B_3   | B_4   | B_5   | delay |
|------|-------|-------|-------|-------|-------|-------|-------|
| 0    | BCD465| BCD465| BCD465| BCD465| BCD465| 5618两门|       |
| 1    | 5618两门| 5618两门| 5618两门| 5618两门| 5618两门| 5618两门|       |
| 2    | BCD530| BCD530| BCD530| BCD500| BCD500| BCD500|       |
| 3    | BCD580| BCD580| BCD580| BCD580| BCD580| BCD580|       |
| 4    | 5620三门| 5620三门| 5620三门| 5620三门|       |       |       |
| 5    | 5620两门| 5620两门| 5620两门| 5620两门|       |       |       |
| 6    | BCD320| BCD320| BCD320| BCD320| BCD320|       |       |
| 7    |       | BCD4501| BCD4501| BCD4501| BCD4501|       |       |
| 8    |       | BCD580| BCD580| BCD580| BCD580|       |       |


