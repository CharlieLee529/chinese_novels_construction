# -*- coding: utf-8 -*-

import os
import time
import uuid
import requests
import sys
import json
import uuid
import urllib3
import random
import time
import copy
from tqdm import tqdm
import traceback
import pandas as pd
# urllib3.disable_warnings()
import logging
# logging.captureWarnings(True)
# requests.packages.urllib3.disable_warnings()
import multiprocessing
import time
import datetime
from dotenv import load_dotenv

import query_merge_all

def get_timestamp(is_short=False):
    if is_short:
        timestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    else:
        timestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S.%f')
    return timestr

def build_task_setting_generate(file_path):
    ret_list = []

    with open(file_path,'r') as f:
        for li in f:
            li = li.strip()
            content = json.loads(li)
            # print(content)
            ret_list.append(content)

    return ret_list



def get_infer_res_only_last(content):
    # constrain="你的回答中最多只能有2个括号文学，总回答长度不超过100个字。"
    ret_c = copy.deepcopy(content)
    if 'system_prompt' in content:
        sp = content['system_prompt'] #+ "\n" + constrain
        messages = [{"role":"system","content":sp}]+content['messages']
    elif content['messages'][0]['role']=='system':
        messages = content['messages']
    else:
        raise Exception("Can Not Found System Prompt!!!")
    
    try:
        response,thinking = ROLE_MODEL.run_model(messages)
        ret_c['model_answer'] = response
        ret_c['model_answer_thinking'] = thinking
    except:
        # import traceback
        # traceback.print_exc()
        ret_c['model_answer'] = 'ERROR'
    
    return ret_c
                

def get_infer_res(content):
    # constrain="你的回答中最多只能有2个括号文学，总回答长度不超过100个字。"
    sp = content.get('system_prompt',"You are a helpful assistant.") #+ "\n" + constrain
    if 'question_list' in content:
        q_list = content['question_list']#[:5]
    else:
        q_list = []
        for msg in content['messages']:
            if msg['role']=='user':
                q_list.append(msg['content'])
    msgs = [{"role":"system","content":[{"type":"text","value":sp}]}]
    answers = []
    for q in q_list:
        msgs.append({
            "role":"user","content":[{"type":"text","value":q}]
        })
        error_cnt = 3
        while error_cnt>0:
            try:
                # print("msgs",msgs)
                response = ROLE_MODEL.run_model(msgs)
                raw_answer = response #response['answer'][-1]['value']
                # print("res:",response)
                msgs.append({
                    "role":"assistant","content":[{"type":"text","value":raw_answer}]
                })
                break
            except:
                error_cnt-=1
        
        if error_cnt<=0:
            msgs.append({
                        "role":"assistant","content":[{"type":"text","value":"ERROR"}]
                    })
    content['new_messages'] = msgs
    return content

class Rolemodel_distll:
    def __init__(self,model_client) -> None:
        self.model_client = model_client
        self.model_name = model_client.model_name
        # self.more_constrain = "\n你的回复长度不超过100个字。"
    
    def get_query_messages(self,content):
        first_sentence = content['messages'][0]['content']
        first_q = content['messages'][1]['content']
        role_model_messages = [
            {"role":"system","content":content['system_prompt']} ,
            {"role":"user","content":first_sentence},
            {"role":"assistant","content":first_q}
        ]
        return role_model_messages
    
    def run_model(self,query_messages):
        # query_messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': '你是谁？'}]
        # print("query_messages",query_messages)

        response = self.model_client.run_model(query_messages)
        # print('response:',response)
        # for doubao-seed-1.6-1015
        pure_response = response['choices'][0]['message']['content']
        pure_thinking = response['choices'][0]['message'].get('reasoning_content',"")

        # for gemini
        # pure_response = response['candidates'][0]['content']['parts'][-1]['text']
        # pure_thinking = ""
        return pure_response,pure_thinking

class Rolemodel_crawl:
    def __init__(self,model_client) -> None:
        self.model_client = model_client
        self.model_name = model_client.model_name
        # self.more_constrain = "\n你的回复长度不超过100个字。"
    
    def get_query_messages(self,content):
        first_sentence = content['messages'][0]['content']
        first_q = content['messages'][1]['content']
        role_model_messages = [
            {"role":"system","content":content['system_prompt']} ,
            {"role":"user","content":first_sentence},
            {"role":"assistant","content":first_q}
        ]
        return role_model_messages
    
    def run_model(self,query_messages):
        response = self.model_client.run_model(query_messages)
        pure_response = response['answer'][0]['value']
        return pure_response

class Rolemodel_taiji:
    def __init__(self,model_client) -> None:
        self.model_client = model_client
        self.model_name = model_client.model_name
        # self.more_constrain = "\n你的回复长度不超过100个字。"
    
    def get_query_messages(self,content):
        first_sentence = content['messages'][0]['content']
        first_q = content['messages'][1]['content']
        role_model_messages = [
            {"role":"system","content":content['system_prompt']} ,
            {"role":"user","content":first_sentence},
            {"role":"assistant","content":first_q}
        ]
        return role_model_messages
    
    def run_model(self,query_messages):
        response = self.model_client.run_model(query_messages)
        # print("response:",response)
        # for normal taiji service
        pure_response = response['choices'][0]['message']['content']
        # for deepseek taiji service
        pure_thinking= response['choices'][0]['message']['reasoning_content']
        return pure_response,pure_thinking

if __name__ == "__main__":

    # 1、蒸馏平台模型
    # load_dotenv()
    # APP_ID=os.getenv("EVL_APPID")
    # APP_KEY=os.getenv("EVL_APPKEY")
    # # llm = "api_doubao_DeepSeek-V3.1-250821"
    # # llm = "api_openai_gpt-5-chat-latest"
    # # llm = "api_openai_gpt-5.1"
    # # llm = "api_google_gemini-2.5-pro"
    # # llm = "api_google_gemini-3-pro-preview"
    # # llm = "api_google_gemini-2.5-pro"
    # # llm = "api_naci_default_gemini-3-pro-preview"
    # # llm = "api_aws_third_anthropic.claude-sonnet-4-5-20250929-v1:0"
    # # llm = "api_openai_chatgpt-4o-latest"
    # # llm = "api_doubao_doubao-seed-1-6-251015"
    # # llm = "api_doubao_Doubao-1.5-pro-32k-character-250715"
    # llm = "api_doubao_deepseek-v3-2-251201"
    # role_model_client = query_merge_all.HunyuanApi("http://{}:8080".format("trpc-gpt-eval.production.polaris"), APP_ID, APP_KEY, llm)
    # ROLE_MODEL = Rolemodel_distll(role_model_client)
    # print('llm:',llm)

    # 2、一站式抓取接口模型
    # "api_key": "bd539f29-1d35-4551-8629-dd79df94b212", "model_name": "api_doubao_doubao-seed-1-6-251015"
    # 抓取接口要调整一下，url 用 http://trpc-utools-crawl.turbotke.production.polaris:8009/
    # MODEL_CONFIG={
    #         "api_key":"bd539f29-1d35-4551-8629-dd79df94b212",
    #         "model_marker":"api_doubao_doubao-seed-1-6-251015",
    #         "name":"api_doubao_doubao-seed-1-6-251015"
    #     }
    # role_model_client = query_merge_all.CrawAPI(MODEL_CONFIG)
    # ROLE_MODEL = Rolemodel_crawl(role_model_client)

    # 3、taiji自己部署模型
    # model_name="DeepSeek-V3.2-A37B-eason"
    # model_name="DeepSeek-V3.2-A37B-bio"
    # model_name="gy-math-deepseek-v3.2"
    
    model_name="hy3.0-a20b-roleplay-sft-exp1-charliecyli-test2"
    role_model_client = query_merge_all.TaijiApi(model=model_name)
    ROLE_MODEL = Rolemodel_taiji(role_model_client)

    eval_data_path = '/apdcephfs_qy4/share_302593112/rolandwu/project/hunyuan_luodi/LLM-RolePlay-R1/eval_data/角色评测/20260228-质检prompt/query_data/get_test_result.jsonl'

    # eval_data_path = '/apdcephfs_qy4/share_302593112/rolandwu/project/hunyuan_luodi/LLM-RolePlay-R1/eval_data/角色评测/20260228-质检prompt/query_data/get_data_quality_check-ds3.2—20260309.jsonl'

    eval_file_name = eval_data_path.split('/')[-1].replace('.json','')
    t1 = time.time()
    task_list = build_task_setting_generate(eval_data_path)
    t2 = time.time()
    print(f"cost {t2-t1} seconds to load files.")

    now_timestr = get_timestamp(is_short=True)

    if not os.path.exists('../query_res/20260309/'):
        os.makedirs('../query_res/20260309/')
    
    MODEL_NAME = ROLE_MODEL.model_name

    save_path = f'../query_res/20260309/{MODEL_NAME}-fast-[{eval_file_name}]-{now_timestr}.output'
    print('save_path',save_path)

    wf = open(save_path,'w')


    task_list = task_list[:10]
    # task_list = task_list * 3

    saving_list = []
    with multiprocessing.Pool(30) as p:
        # for idx,ts in enumerate(tqdm(p.imap(get_infer_res,task_list),total=len(task_list))):
        for idx,ts in enumerate(tqdm(p.imap(get_infer_res_only_last,task_list),total=len(task_list))):
            ts['question_idx'] = idx
            ts[f'query_model_{now_timestr}'] = ROLE_MODEL.model_name
            saving_list.append(ts)
            wf.write(json.dumps(ts,ensure_ascii=False) + '\n')
            if len(saving_list)%100==0:
                print('saving_list:',len(saving_list))
                wf.flush()