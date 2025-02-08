from ai_flask_web.platform_app import PlatformApp, request, logger
from flask import jsonify
import logging
import json
import time
from mold_layout.mould_arrangement_solver import MouldArrangementSolver
from mold_layout.config_loader import get_host, get_access_token
solver = MouldArrangementSolver()

platform_app = PlatformApp(solver)
app = platform_app.app
app.config['JSON_AS_ASCII'] = False

@app.post("/solve-mould-arrangement")
def solve_mould_arrangement():
    data = request.get_json()

    # order_info = json.loads(data.get('order_info'))
    # fixed_info = json.loads(data.get('fixed_info'))
    order_info = data.get('order_info')
    work_calendar_id = data.get('work_calendar_id')


    # host = data.get('host')
    # access_token = data.get('access_token')
    host = get_host()
    access_token = get_access_token()

    res = {'code': 200, 'message': None, 'data': None}
    start_time = time.time()
    try:
        solver.solve(order_info, work_calendar_id, host, access_token)
        solver.delete_intermediate_files()
        result_a = solver.result_a
        result_b = solver.result_b
        res['data'] = {'result_a': result_a, 'result_b': result_b}
    except Exception as e:

        logger.error(e, exc_info=True, stack_info=True)
        res['code'] = 500
        res['message'] = str(e)
        raise e

    res_str = json.dumps(res, ensure_ascii=False)
    logger.info("url:[%s],use[%s]ms,body:[%s],res:[%s]", request.url, str((time.time() - start_time) * 1000),
                json.dumps(data, ensure_ascii=False), res_str)

    return jsonify(res)


if __name__ == '__main__':
    app.run()


if __name__ != "__main__":
    gunicorn_logger = logging.getLogger('gunicorn.error')
    for handler in gunicorn_logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
