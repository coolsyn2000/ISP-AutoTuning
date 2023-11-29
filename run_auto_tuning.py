import openbox

from openbox import Optimizer
from openbox import space as sp
import matplotlib.pyplot as plt
import os
import os.path as op

import cv2
import numpy as np
import skimage.io
from Image_quality_assessment.FR_IQA import loss_lpips
from fast_OpenISP.pipeline import Pipeline
from fast_OpenISP.utils.yacs import Config
import yaml

OUTPUT_DIR = './fast_OpenISP/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def objective_function(config):
    with open('./fast_OpenISP/configs/test.yaml', 'r') as file:
        yaml_data = file.read()

    # 将YAML数据解析为字典
    data_dict = yaml.safe_load(yaml_data)
    data_dict['nlm']['h'] = config['x1']
    data_dict['eeh']['edge_gain'] = config['x2']
    with open('./fast_OpenISP/configs/test.yaml', 'w') as file:
        file.write(yaml.dump(data_dict, allow_unicode=True, default_flow_style=False, sort_keys=False))

    cfg = Config(data_dict)
    pipeline = Pipeline(cfg)


    raw_path = './fast_OpenISP/raw/test.RAW'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))

    data, _ = pipeline.execute(bayer)

    output_path = op.join(OUTPUT_DIR, 'test.png')
    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)

    obj_func = loss_lpips('./fast_OpenISP/output/ref.png','./fast_OpenISP/output/test.png',net='alex')

    return obj_func

if __name__ == '__main__':
    # Define Search Space
    space = sp.Space()
    x1 = sp.Int("x1", 0, 300, default_value=150)
    x2 = sp.Int("x2", 0, 4096, default_value=2048)
    space.add_variables([x1, x2])

    opt = Optimizer(
        objective_function,
        space,
        max_runs=50,
        surrogate_type='prf',
        task_id='auto_tuning',
        # Have a try on the new HTML visualization feature!
        # visualization='advanced',   # or 'basic'. For 'advanced', run 'pip install "openbox[extra]"' first
        # auto_open_html=True,        # open the visualization page in your browser automatically
    )

    history = opt.run()

    # print(history)

    history.plot_convergence(true_minimum=0.397887)
    plt.show()