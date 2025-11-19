import os
configs_list = [
    # rs semantic segmentation
    './configs/cfg_openearthmap.py',
    './configs/cfg_loveda.py',
    './configs/cfg_iSAID.py',
    './configs/cfg_potsdam.py',
    './configs/cfg_vaihingen.py',
    './configs/cfg_uavid.py',
    './configs/cfg_udd5.py',
    './configs/cfg_vdd.py',
    # # rs single-class
    # './configs/cfg_whu_aerial.py',
    # './configs/cfg_whu_sat_II.py',
    # './configs/cfg_inria.py',
    # './configs/cfg_xBD.py',
    # './configs/cfg_chn6-cug.py',
    # './configs/cfg_deepglobe_road.py',
    # './configs/cfg_massachusetts_road.py',
    # './configs/cfg_spacenet_road.py',
    # './configs/cfg_wbs-si.py',
]

# for config in configs_list:
#     print(f"Running {config}")
#     # os.system(f"bash ./dist_test.sh {config}")
#     os.system(f'python eval.py --config {config} --work-dir work_dirs/tmp')
output_file = './log/2gpu_jbu_one_forward_x.log'
# output_file = './log/clip_v_jbu_one.log'
with open(output_file, 'w') as log_file:
        for config in configs_list:
            print(f"Running {config}")
            # 将输出同时写入到日志文件
            log_file.write(f"Running {config}\n")
            # os.system(f'python eval.py --config {config} --work-dir work_dirs/tmp')
            # 执行命令并将输出重定向到对应的文件
            command = f'python eval.py --config {config} --work-dir work_dirs/tmp'


            os.system(f'{command} >> {output_file} 2>&1')  # 追加输出到对应日志文件