import json
import re
import yaml
import os
import glob
import copy
import string
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime

from collections import defaultdict

def get_timestamp():
    timestr = datetime.datetime.now().strftime('%Y%m%d')
    return timestr

def load_res(file_path):
    ret_list = []
    source = file_path.split('/')[-1]
    with open(file_path,'r') as f:
        for li in f:
            # print(li)
            li = li.strip()
            content = json.loads(li)
            content['temp_source_file'] = source
            # try:
            #     content['question'][0] = content['question'][0].split('<unused5>')[-1]
            # except Exception as e:
            #     print(e)
                
            ret_list.append(content)
    # print(len(ret_list))
    return ret_list

def build_query_dict(query_str,sp=None):
    ret_dict = {
        "use_openai_format": 1,
        "system_prompt":"You are a helpful assistant.",
        "messages":[
            # {
            #     "role":"system",
            #     "content":"You are a helpful assistant." if not sp else sp
            # },
            {
                "role":"user",
                "content":query_str
            }
        ],
        "parameters":{
            "temperature":0.7,
            "top_p":0.6,
            "top_k":20,
            "do_sample": False,
        },
    }
    return ret_dict


def save_to_jsonfile(org_data,file_name):
    print(f"total_lines:{len(org_data)}")
    with open(file_name,'w') as f:
        for content in org_data:
            f.write(json.dumps(content,ensure_ascii=False)+'\n')

def md5_convert(string):
    """
    计算字符串md5值
    :param string: 输入字符串
    :return: 字符串md5
    """
    m = hashlib.md5()
    m.update(string.encode())
    return m.hexdigest()

def msg_to_str(msgs,user_name='用户',model_name='回复'):
    ret_list = []
    for msg in msgs:
        c = msg['content']
        if isinstance(c,list):
            c = c[-1]['value']
        if msg['role']=='user':
            ret_list.append(f"{user_name}:{c}")
        elif msg['role']=='assistant':
            ret_list.append(f"{model_name}:{c}")
    return "\n".join(ret_list)
    
def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

def merge_prompt(hit_cid,cid_dict):
    k1 = 'message_pair' #'messages'

    saving_list = []

    for cid in hit_cid:
        # content = cid_dict[cid][0]
        processed = []
        final_msgs = []
        like_response = []
        unlike_response = []
        temp_c = copy.deepcopy(cid_dict[cid][0])
        
        del temp_c[k1]
        if 'req_messages' in temp_c:
            del temp_c['req_messages']

        for content in cid_dict[cid]:
            if content['like_cnt']>0:
                like_response.append(content[k1])
            if content['unlike_cnt']>0:
                unlike_response.append(content[k1])
            
            last_q = ""
            q_list = []
            a_list = []
            for idx,msg in enumerate(content[k1]):
                msg_c = msg.get("content","")
                if idx%2==0:
                    q_list.append(msg_c)
                else:
                    a_list.append(msg_c)
            assert len(q_list) == len(a_list)

            
            for q,a in zip(q_list,a_list):
                if q+a in processed:
                    continue

                if len(final_msgs)==0:
                    final_msgs.append({"role":"user","content":q})
                    final_msgs.append({"role":"assistant","content":a})
                else:
                    if final_msgs[-2]['content'] == q:
                        old_a = final_msgs[-1]['content']
                        final_msgs[-1]['content'] = a
                        oc = final_msgs[-1].get("old_content",[])
                        oc.append(old_a)
                        final_msgs[-1]['old_content'] = oc
                    else:
                        final_msgs.append({"role":"user","content":q})
                        final_msgs.append({"role":"assistant","content":a})
                processed.append(q+a)
        
            # print(len(final_msgs))
        temp_c[k1] = final_msgs
        temp_c['like_msgs'] = like_response
        temp_c['unlike_response'] = like_response
        saving_list.append(temp_c)
    return saving_list

def count_length(c_list):
    ret_int = 0
    for c in c_list:
        ret_int += len(c['messages'])
    return ret_int

if __name__ == "__main__":
    dir_path = '/apdcephfs_qy4/share_302593112/rolandwu/project/hunyuan_luodi/LLM-RolePlay-R1/train_data/快慢思考数据班车/20251029-本地实验-快思考/角色扮演/日志数据/from_alice/20251020_high_turn_dsv3_filter_v4.prompt_class/*.prompt_class'
    dir_path = '/apdcephfs_qy4/share_302593112/rolandwu/project/hunyuan_luodi/LLM-RolePlay-R1/train_data/快慢思考数据班车/20251029-本地实验-快思考/角色扮演/日志数据/from_alice/20251021_high_turn_dsv3_filter_v4.prompt_class/*.prompt_class'
    # dir_path = "/apdcephfs_cq8/share_1324356/alicexfeng/get_text_creation_log/20251022_5_turn_dsv3_filter.prompt_class/*.prompt_class"
    # dir_path = "/apdcephfs_cq8/share_1324356/alicexfeng/get_text_creation_log/20251023_5_turn_dsv3_filter.prompt_class/*.prompt_class"
    # dir_path = "/apdcephfs_cq8/share_1324356/alicexfeng/get_text_creation_log/20251110_5_turn_dsv3_filter.prompt_class/*.prompt_class"
    # dir_path = "/apdcephfs_cq8/share_1324356/alicexfeng/get_text_creation_log/20251111_5_turn_dsv3_filter.prompt_class/*.prompt_class"
    # dir_path = "/apdcephfs_cq8/share_1324356/alicexfeng/get_text_creation_log/20251112_5_turn_dsv3_filter.prompt_class/*.prompt_class"
    # dir_path = '/apdcephfs_cq8/share_1324356/alicexfeng/get_text_creation_log/20251213.prompt_class/*.prompt_class'
    # dir_path = '/apdcephfs_cq8/share_1324356/alicexfeng/get_text_creation_log/20251214.prompt_class/*.prompt_class'
    # dir_path = '/apdcephfs_cq8/share_1324356/zhangyusong/chat_with_model_topic/worktable/0123_hunyuan_vs_ds_ab/20251229_20260117_mas_intent_*/*.json'
    dir_path = '/apdcephfs_cq8/share_1324356/zhangyusong/chat_with_model_topic/worktable/0123_hunyuan_vs_ds_ab/20251229_20260117_mas_intent_3/*.json'
    # dir_path = '/apdcephfs_qy4/share_302593112/easonszhang/ab_data_export/0123_ds_vs_hunyuan_v2/*.json'

    print("dir_path",dir_path)

    dir_name = dir_path.split('/')[-2]

    data_path_list = glob.glob(dir_path)
    data_path_list = sorted(data_path_list)
    print("total_file:",len(data_path_list))

    start_idx = 0
    end_idx = 200#1596
    res_alice = []
    for path in tqdm(data_path_list[start_idx:end_idx]):
        # 1、加载数据
        temp_data = load_res(path)
        res_alice += temp_data


    # 2、按照cid合并
    cid_dict = defaultdict(list)
    all_messages = 0
    for content in res_alice:
        cid = content['cid']
        cid_dict[cid].append(content)
        all_messages += len(content['message_pair'])

    print("total_session",len(cid_dict),all_messages)

    # 3、筛选角色扮演相关的cid
    hit_cid = []
    rm = 0
    first_line = []
    for k,v in cid_dict.items():
        # v_sorted = sorted(v,key=lambda x:x['promptcreatetime'])
        v_sorted = sorted(v,key=lambda x:x['exp_time'])

        first_line.append({
            "sp":v_sorted[0]['message_pair'][0]['content']
        })

        if '扮演' in v_sorted[0]['message_pair'][0]['content']:
            hit_cid.append(k)
            for c in cid_dict[k]:
                rm += len(c['message_pair'])

    # save_to_jsonfile(first_line,f"./raw_data/get_sp_classify_{dir_name}_{start_idx}_{end_idx}.jsonl")
    print("roleplay session",len(hit_cid),rm)

    # 4、整合数据
    saving_list = merge_prompt(hit_cid,cid_dict)

    total_line = sum([len(a['message_pair']) for a in saving_list])


    print('saving_list:',len(saving_list),'turn:', total_line)

    saving_list_filter = []
    for content in saving_list:
        # msg_str = msg_to_str(content['messages'])
        msg_str = msg_to_str(content['message_pair'])

        msg_str = msg_str.lower()
        if 'deepseek' not in msg_str and '元宝' not in msg_str:
            saving_list_filter.append(content)

    print(len(saving_list_filter))

    save_to_jsonfile(saving_list_filter,f'./raw_data/candidate_eval_filter_{dir_name}_{start_idx}_{end_idx}.jsonl')
