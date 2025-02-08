import os


def get_host():
    host = os.getenv('DOCS_SERVICE_HOST', 'http://docs-service.qd-aliyun-dmz-ack-internal.haier.net')
    return host


def get_access_token():
    access_token = os.getenv('DOCS_SERVICE_ACCESS_TOKEN', '68f5ff02-4250-4a5a-a51c-bc0d911f1e2d')
    return access_token
