# -*- coding: utf-8 -*-
# @Time    : 2021/5/13 8:29 下午
# @Author  : shaoguowen
# @Email   : shaoguowen@tencent.com
# @FileName: censeoqoe_platform_to_json.py
# @Software: PyCharm

"""
基于censeo平台标注的数据，转换成censeo-train需要统一训练json格式
"""

import os
import json
from multiprocessing import Pool
import re
from collections import defaultdict
import numpy as np
import cv2
from utils import utils


def video2imgs(v_dir, vname, save_vdir):
    """
    将视频extract帧并保存
    :param v_dir:
    :param vname:
    :param save_vdir:
    :return:
    """
    vpath = os.path.join(v_dir, "pgc_videos_mark_clip", vname)
    if not os.path.exists(vpath):
        vpath = os.path.join(v_dir, "pgc_videos_mark_clip_v3", vname)
    vgen = utils.read_video_gen(vpath, 0, -1, 10, 1)
    save_vsubdir = os.path.join(save_vdir, vname)
    os.makedirs(save_vsubdir, exist_ok=True)
    i = 1
    for frames in vgen:
        print("save image at:{}".format(os.path.join(save_vsubdir, "%05d.png" % i)))
        cv2.imwrite(os.path.join(save_vsubdir, "%05d.png" % i), frames[0])
        i += 1


def process(json_in_path, json_out_path):
    """
    处理censeo平台标注得到的数据
    :param json_in_path: censeo平台输出的打分json文件
    :param json_out_path: 处理后的json文件，可用于censeo的直接训练
    :return:
    """
    with open(json_in_path, 'r') as fin:
        data = json.load(fin)

    media_map = {str(info["id"]): info["original_name"] for info in data['media']}
    result = defaultdict(list)
    for media_info in data['scores']:
        for i in range(len(media_info['annotation_task'][0]['scores'])):
            score = media_info['annotation_task'][0]['scores'][i]['score']
            if score == 0 or media_info['annotation_task'][0]['scores'][i]['annotator_id'] in [22, 29]:
                print(media_info['annotation_task'][0]['scores'][i]['annotator_id'])
                continue
            result[media_map[media_info['media_id']]].append(int(score))
    print(len(result))
    results = {}
    vnames_ref_set = set()
    for vname in result:
        mos_gt = result[vname]
        ref_vname = vname.split(".mp4")[0] + ".mp4-" + re.findall("-clip(\d+)", vname)[0]  # 这里要根据实际情况改
        vnames_ref_set.add(ref_vname)
        info_r = {'ref_video': ref_vname, 'ref_dir': '',
                  'dis_dir': '', 'mos': np.mean(mos_gt),
                  'mos_dv': None, 'std': np.std(mos_gt), 'mos_list': mos_gt}

        results[vname] = info_r
    print(len(vnames_ref_set))
    with open(json_out_path, "w") as fw:
        json.dump(results, fw)
    return results


if __name__ == '__main__':
    VIDEO_ORG_DIR = "/media/shaoguowen/shaoguo/IQA/PGC-compressed-videos"
    JSON_IN_PATH = './data/pgc/pgc_video_acr_censeo_0420.json'
    JSON_OUT_PATH = './data/pgc/pgc_video_acr_censeo_0420_processed.json'
    results = process(JSON_IN_PATH, JSON_OUT_PATH)

    # 视频处理才需要extract视频帧，否则注释以下的代码，不用执行video2imgs
    VIDEO_FRAMES_SAVE_DIR = "/media/shaoguowen/shaoguo/IQA/PGC-compressed-videos_frames"
    os.makedirs(VIDEO_FRAMES_SAVE_DIR, exist_ok=True)
    pool = Pool(8)
    for vname in results:
        pool.apply_async(video2imgs, args=(VIDEO_ORG_DIR, vname, VIDEO_FRAMES_SAVE_DIR))
    pool.close()
    pool.join()
