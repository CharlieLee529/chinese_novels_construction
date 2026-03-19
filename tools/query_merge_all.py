import json
import logging
import time
import datetime
import base64
import hmac
import hashlib
import requests
import random
import uuid
import sseclient
import traceback
from tqdm import tqdm

API_VERSION = "v2.03"

import os
from dotenv import load_dotenv

load_dotenv()

APP_ID=os.getenv("EVL_APPID")
APP_KEY=os.getenv("EVL_APPKEY")


SS_URL="http://stream-server-online-sbs-10103.turbotke.production.polaris:81/openapi/chat/completions"
# SS_URL="http://stream-server-online-openapi.turbotke.production.polaris:1081/openapi/chat/completions"

class TaijiApi:
    """
        用于请求一站式自己部署的模型
    """
    def __init__(self,  model, wsid="10103",ss_url=SS_URL):
        self.ss_url = ss_url
        self.timeout = 3600  # 超时时间
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 7auGXNATFSKl7dF",
            "Wsid": wsid
        }
        self.model = model
        self.model_name = model

    def run_model(self, messages, temperature=0.0):
        """_summary_

        Args:
            messages (_type_): [
                    {"role":"system","content":"你是李白"},
                    {"role":"user","content":"你好吗"},
                    {"role":"assistant","content":"我是李白，还不错，你呢"},
                    {"role":"user","content":"凑合"}
                ]
            temperature (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        enable_stream = False
        data = {
            "model": self.model,
            "query_id": "test_query_id_" + str(uuid.uuid4()),
            "messages": messages,
            "stream": enable_stream,
            "temperature": temperature,
            "use_t1_client": True,
            "output_seq_len": 32768,
            "max_input_seq_len": 65536,
        }

        try:
            rsp = requests.post(self.ss_url, headers=self.headers, json=data, stream=True)
            if enable_stream:
                client = sseclient.SSEClient(rsp)
                answer = ''
                for event in client.events():
                    if event.data != '':
                        data_js = json.loads(event.data)
                        try:
                            answer += data_js['choices'][0]['delta']['content']
                        except:
                            pass
                assert data_js['choices'][0]['finish_reason'] == 'stop'
                return {'choices': [{'message': {'content': answer, 'reasoning_content': ''}}]}
            else:
                # assert rsp.json()['choices'][0]['finish_reason'] == 'stop'
                return rsp.json()
        except Exception:
            import traceback
            traceback.print_exc()
            return {"error": "request llm failed"}

class CrawAPI:
    """
        参考：用来抓取一站式-数据抓取服务中，申请的服务的脚本
        MODEL_CONFIG={
            "api_key":"bd539f29-1d35-4551-8629-dd79df94b212",
            "model_marker":"api_doubao_doubao-seed-1-6-251015",
            "name":"api_doubao_doubao-seed-1-6-251015"
        }
    """
    def __init__(self, model_config):
        self.model_config = model_config
        self.model_name = model_config['name']
    
    def run_model(self, messages):
        """_summary_

        Args:
            messages [
                    {"role":"system","content":"你是李白"}
                    {"role":"user","content":"你是谁"},
                    {"role":"assistant","content":"我是李白，你呢"},
                    {"role":"user","content":"我是李黑"}
                ]
            messages输入都是这个格式，具体的格式适配由本地完成
        Returns:
            _type_: _description_
        """
        if messages[0]['role']=='system':
            system_prompt = messages[0]['content']
            messages.pop(0)
        else:
            system_prompt = ""

        tmp_msgs = []
        for msg in messages:
            if isinstance(msg['content'],str):
                msg['content'] = [{
                    "type":"text",
                    "value":msg['content']
                }]
            tmp_msgs.append(msg)
        assert len(tmp_msgs)==len(messages)

        headers = {
            'Content-Type': 'application/json',
        }
        # url = "http://trpc-utools-prod.turbotke.production.polaris:8009/"
        url = "http://trpc-utools-crawl.turbotke.production.polaris:8009/"
        timeout=300
        data = {
            "bid": "open_api_test", # 固定值
            "server": "open_api", # 固定值
            "services": [], # 固定值
            "bid_2": "B端", # 二级业务：B端 or C端
            "bid_3": "产品A", # 三级业务：eg：腾讯会议、qq浏览器
            "request_id": str(uuid.uuid4()) + "rolandwu",  # 必传值，请求id 使用uuid+username
            "session_id": self._get_timestamp(),  # 必传值，会话id
            "api_key": self.model_config['api_key'],  # 必传值，判断配额,需要用户申请需求单进行更换
            "model_marker":self.model_config['model_marker'],# 查看文档中model_marker
            "system": system_prompt, # 模型人设
            "messages": tmp_msgs, # 历史轮次和当前轮次的会话
            "params":{"thinking":{"type":"disabled"}},
            "general_params": {"thinking":{"type":"disabled"}}, # 竞品大模型通用超参，优先级高于params，降低业务方适配成本
            "timeout": timeout, # ，调模型接口的超时时间,单位秒
            "extension": {}, # ，扩展字段，非必填
            "model_name":self.model_config['name']
        }

        # print(f"data:{data}")

        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
        ret_res = response.json()

        return ret_res

    def _get_timestamp(self):
        timestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S.%f')
        return timestr

def get_simple_auth(source, SecretId, SecretKey):
    dateTime = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    auth = "hmac id=\"" + SecretId + "\", algorithm=\"hmac-sha1\", headers=\"date source\", signature=\""
    signStr = "date: " + dateTime + "\n" + "source: " + source
    sign = hmac.new(SecretKey.encode(), signStr.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    sign = auth + sign + "\""
    return sign, dateTime

class HunyuanApi:
    """
        请求蒸馏平台
    """
    def __init__(self, host, user, apikey, model):
        self.host = host
        self.user = user
        self.apikey = apikey
        self.timeout = 3600  # 超时时间
        self.model = model
        self.user = APP_ID
        self.apikey = APP_KEY
        self.sp = ""
        self.model_name = model

    def get_header(self):
        source = 'agent_trajectory_gen'  # 签名水印值，可填写任意值
        sign, dateTime = get_simple_auth(source, self.user, self.apikey)
        headers = {'Apiversion': API_VERSION, 'Authorization': sign, 'Date': dateTime, 'Source': source}
        return headers

    def run_model(self, raw_messages, use_stop=False):
        """
        

        Args:
            messages (_type_): _description_
                [
                    {"role":"system","content":"你是李白"}
                    {"role":"user","content":"你是谁"},
                    {"role":"assistant","content":"我是李白，你呢"},
                    {"role":"user","content":"我是李黑"}
                ]
            use_stop (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        messages = raw_messages
        if messages[0]['role']=='system':
            self.sp = messages[0]['content'] if isinstance(messages[0]['content'],str) else messages[0]['content'][0]['value']
            messages.pop(0)
            
        tmp_msgs = []
        for msg in messages:
            if isinstance(msg['content'],str):
                msg['content'] = [{
                    "type":"text",
                    "value":msg['content']
                }]
            tmp_msgs.append(msg)
        assert len(tmp_msgs)==len(messages)

        # print("messages",messages)
        
        base_url = self.host + '/api/v1/data_eval'
        model_marker = self.model
        model_name = self.model

        if use_stop:
            data = {
                "request_id": str(uuid.uuid4()),
                "model_marker": model_marker,
                "system":self.sp,
                "messages": messages,
                "params": {
                    "model": model_name,
                    "stop_sequences": ["\n<tool_response>", "<tool_response>"],
                },
                "timeout": 1200
            }
        else:
            data = {
                "request_id": str(uuid.uuid4()),
                "model_marker": model_marker,
                "system":self.sp,
                "messages": messages,
                "params": {
                    "model": model_name,
                },
                "timeout": 1200
            }
            
        if 'V3.1' in model_marker or 'Seed' in model_marker:
            data["params"]["thinking"] = {"type": "enabled"}
        elif 'gemini' in model_marker.lower():
            data["params"]["generationConfig"] = {
                "thinkingConfig":{
                                    "include_thoughts": True
                                }
                 }
        elif 'claude' in model_marker:
            del data['params']['model']
            data['params']['thinking'] = {
                "type":"enabled",
                "budget_tokens": 4096
            }
        elif model_marker=='api_doubao_doubao-seed-1-6-251015':
            # data["params"]["thinking"] = {"type": "disabled"}
            data["params"]["thinking"] = {"type": "enabled"}
            data['params']['reasoning_effort'] = "high" #minimal、low、medium、high

        
        # del data['system']
        # print("data",json.dumps(data,ensure_ascii=False))
        
        
        # if system_prompt:
        #     data['system'] = system_prompt

        headers = dict(self.get_header())
        try:
            rsp = requests.post(url=base_url, headers=headers, json=data, timeout=self.timeout)
            # print(f"rsp.json():{rsp.json()}")
            if "qwen3-max-preview" in model_marker:
                return rsp.json()
            else:
                # print("ret_type",type(rsp.json()["request_detail"]["response"]))
                # print("resp.json.request_detaul.response:",rsp.json()["request_detail"]["response"])
                # print('rsp.json():',rsp.json())
                # print("messages:",messages)
                return rsp.json()["request_detail"]["response"]
        except Exception:
            # import traceback
            # traceback.print_exc()
            # print(f"error_input:{raw_messages}")
            return {"error": "request llm failed"}

if __name__ == "__main__":

    test_messages = [
                    {"role":"system","content":"你是李白"},
                    {"role":"user","content":"你是谁"},
                    {"role":"assistant","content":"我是李白，你呢"},
                    {"role":"user","content":"我是李黑"}
                ]
    # test_messages = [{'role': 'user', 'content': [{'type': 'text', 'value': '假如你是电影里面的郭靖，郭靖从小跟随父母流浪江湖，后被黄蓉收养，成为黄药师的弟子，习得武艺，与黄蓉共同成长，最终成为一代英雄,下面是你的性格：忠诚正直、善良、勇敢、有担当,厌恶厌恶邪恶、背信弃义的人.已知你精通射雕三部曲武学，包括降龙十八掌、空灵无物、神行百变等.你爱好喜欢大自然，尤其是山林和动物.郭靖喜欢喜欢帮助别人，尤其是弱者.已知你黄蓉：挚爱的妻子，互相扶持.已知你欧阳锋：曾敌后友，共同抵抗外敌.你的做事风格是果断、勇敢、负责任.已知你擅长使用弓箭，射程远、准确度高.你常说的一句话是侠之大者，为国为民.已知你洪七公：师父，教导武艺.已知你杨康：曾经的朋友，后因背叛而反目.厌恶厌恶欺凌弱小、作恶多端的行为,对话的时候你有下面的特征语气古风，谦逊有礼,你对自己称呼为在下郭靖，你对别人为贵人\n示例问答\n问： Hi，可以认识一下吗？我叫于洋。\n答：贵人于洋，郭靖有幸相识，愿与您结交为友。'}]}]

    # 1、测试蒸馏平台
    # llm = "api_doubao_DeepSeek-V3.1-250821"
    # llm = "api_openai_chatgpt-4o-latest"
    # client = HunyuanApi("http://{}:8080".format("trpc-gpt-eval.production.polaris"), APP_ID, APP_KEY, llm)
    # res = client.run_model(test_messages)
    # print(res)

    # 2、测试抓取平台
    # "api_key": "bd539f29-1d35-4551-8629-dd79df94b212", "model_name": "api_doubao_doubao-seed-1-6-251015"
    # 抓取接口要调整一下，url 用 http://trpc-utools-crawl.turbotke.production.polaris:8009/
    # MODEL_CONFIG={
    #         "api_key":"bd539f29-1d35-4551-8629-dd79df94b212",
    #         "model_marker":"api_doubao_doubao-seed-1-6-251015",
    #         "name":"api_doubao_doubao-seed-1-6-251015"
    #     }
    # client = CrawAPI(MODEL_CONFIG)
    # res = client.run_model(test_messages)
    # print(res)

    # 3、测试一站式自己部署模型
    # model_name="rolplay_usermodel_qwen2.5-32b-exp1"
    model_name="A32B_15.6T256k_turbos_hunyuan2.0_sft_exp_1029_bus_step3192"
    client = TaijiApi(model=model_name)
    res = client.run_model(test_messages)
    print(res)
    