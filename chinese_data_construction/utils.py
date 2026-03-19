import pdb 
import os
import re 
import random 
try:
	import openai
except ImportError:
	openai = None
import json
import logging
import time  
import pickle
import random
try:
	import tiktoken
except ImportError:
	tiktoken = None
import __main__
from typing import Dict, List

_default_config_path = os.path.join(os.path.dirname(__file__), 'config.json')
_legacy_config_path = '/apdcephfs_qy4/share_302593112/charliecyli/CoSER/data_construction/config.json'
_config_path = _default_config_path if os.path.exists(_default_config_path) else _legacy_config_path
with open(_config_path, 'r', encoding='utf-8') as f:
	config = json.load(f)

streaming = False


def _resolve_log_file(log_file):
	run_output_dir = os.environ.get('CHINESE_DC_RUN_OUTPUT_DIR')
	if not run_output_dir:
		main_args = getattr(__main__, 'args', None)
		run_output_dir = getattr(main_args, 'output_dir', None)

	if run_output_dir and not os.path.isabs(log_file):
		return os.path.join(run_output_dir, os.path.basename(log_file))

	return log_file

def setup_logger(name, log_file, level=logging.INFO, quiet=False):
	logger = logging.getLogger(name)
	logger.setLevel(level)
	log_file = _resolve_log_file(log_file)
	log_dir = os.path.dirname(log_file)
	if log_dir:
		os.makedirs(log_dir, exist_ok=True)

	if logger.hasHandlers():
		logger.handlers.clear()

	file_handler = logging.FileHandler(log_file, encoding='utf-8')
	file_handler.setLevel(logging.DEBUG)
	file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler.setFormatter(file_formatter)
	logger.addHandler(file_handler)

	if not quiet:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
		console_handler.setFormatter(console_formatter)
		logger.addHandler(console_handler)

	return logger

logger = setup_logger(__name__, f'{__file__.split(".")[0]}.log', level=logging.INFO, quiet=False)

from contextlib import contextmanager
import tempfile
@contextmanager
def _tempfile(dir=None,*args, **kws):
	""" Context for temporary file.
	Will find a free temporary filename upon entering
	and will try to delete the file on leaving
	Parameters
	----------
	suffix : string
		optional file suffix
	dir : string
		directory to create temp file in, will be created if doesn't exist
	"""
	if dir is not None:
		os.makedirs(dir, exist_ok=True)
		
	fd, name = tempfile.mkstemp(dir=dir, *args, **kws)
	os.close(fd)
	try:
		yield name
	finally:
		try:
			os.remove(name)
		except OSError as e:
			if e.errno == 2:
				pass
			else:
				raise e
			
@contextmanager
def open_atomic(filepath, *args, **kwargs):
	""" Open temporary file object that atomically moves to destination upon
	exiting.
	Allows reading and writing to and from the same filename.
	Parameters
	----------
	filepath : string
		the file path to be opened
	fsync : bool
		whether to force write the file to disk
	kwargs : mixed
		Any valid keyword arguments for :code:`open`
	"""
	fsync = kwargs.pop('fsync', False)

	original_permissions = os.stat(filepath).st_mode if os.path.exists(filepath) else None 

	with _tempfile(dir=os.path.join(os.path.dirname(filepath), 'temp')) as tmppath:
		with open(tmppath, *args, **kwargs) as f:
			yield f
			if fsync:
				f.flush()
				os.fsync(f.fileno())
		os.rename(tmppath, filepath)
		if original_permissions is not None:
			os.chmod(filepath, original_permissions)

import datetime
def convert_to_timestamp(time_str: str):
	return time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple())

def safe_pickle_dump(obj, fname):
	"""
	prevents a case where one process could be writing a pickle file
	while another process is reading it, causing a crash. the solution
	is to write the pickle file to a temporary file and then move it.
	"""
	with open_atomic(fname, 'wb') as f:
		pickle.dump(obj, f, -1) # -1 specifies highest binary protocol


ERROR_SIGN = '[ERROR]'

cache_path = '.cache.pkl'
cache_sign = True
cache = None
reload_cache = False

def set_cache_path(new_cache_path):
	global cache_path
	cache_path = new_cache_path
	global reload_cache
	reload_cache = True

def cached(func):
	def wrapper(*args, **kwargs):		
		# extract_from_chunk 
		if func.__name__ == 'extract_from_chunk':
			key = ( func.__name__, args[0]['title'], args[1]) 
		else:
			key = ( func.__name__, str(args), str(kwargs.items())) 

		global cache
		global reload_cache

		if reload_cache:
			cache = None # to reload
			reload_cache = False

		if cache == None:
			if not os.path.exists(cache_path):
				cache = {}
			else:
				try:
					cache = pickle.load(open(cache_path, 'rb'))  
				except Exception as e:
					# logger.info cache_path and throw error
					logger.error(f'Error loading cache from {cache_path}')
					cache = {}

		if (cache_sign and key in cache) and not (cache[key] is None):
			return cache[key]
		else:		
			result = func(*args, **kwargs)
			if result != None:
				cache[key] = result
				safe_pickle_dump(cache, cache_path)
			return result

	return wrapper

enc = None

def _get_enc():
	global enc
	if enc is None:
		try:
			if tiktoken is None:
				raise ImportError("tiktoken is not installed")
			enc = tiktoken.get_encoding("cl100k_base")
		except Exception as e:
			logger.warning(f"Failed to load tiktoken encoding: {e}. Using fallback (len/4) for token counting.")
			enc = False
	return enc

def encode(text):
	e = _get_enc()
	if e:
		return e.encode(text)
	# Fallback: rough estimate ~4 chars per token
	return [0] * (len(text) // 4)

def decode(tokens):
	e = _get_enc()
	if e:
		return e.decode(tokens)
	return ""

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
	e = _get_enc()
	if e:
		num_tokens = len(e.encode(string))
	else:
		num_tokens = len(string) // 4
	logger.info(f"Number of tokens: {num_tokens}")
	return num_tokens

@cached
def get_response(model, messages, nth_generation=0, **kwargs):
	# if messages is str
	if isinstance(messages, str):
		messages = [{"role": "user", "content": messages}]

	# Taiji API requires messages array size > 1, prepend a system message if needed
	if config.get('taiji_wsid') and len(messages) == 1:
		messages = [{"role": "system", "content": "You are a helpful assistant."}] + list(messages)

	try:
		import openai

		# Support taiji backend extra headers
		extra_headers = {}
		if config.get('taiji_wsid'):
			extra_headers["Wsid"] = config['taiji_wsid']

		logger.debug(f"[get_response] model={model}, base_url={config['base_url']}, taiji_wsid={config.get('taiji_wsid')}")
		logger.debug(f"[get_response] extra_headers={extra_headers}")
		logger.debug(f"[get_response] messages count={len(messages)}, first msg role={messages[0]['role']}, content length={len(messages[0]['content'])}")

		client = openai.OpenAI(
			api_key=config['api_key'],
			base_url=config['base_url'],
			timeout=180,
			default_headers=extra_headers if extra_headers else None
		)

		# Taiji backend requires extra body fields
		extra_body = {}
		if config.get('taiji_wsid'):
			import uuid
			extra_body = {
				"query_id": f"coser_{uuid.uuid4()}",
				"use_t1_client": True,
				"output_seq_len": 32768,
				"max_input_seq_len": 65536,
			}

		if model.startswith('claude'):
			max_tokens = 8192
		else:
			max_tokens = 16384

		extra_body_arg = extra_body if extra_body else openai.NOT_GIVEN
		logger.debug(f"[get_response] max_tokens={max_tokens}, temperature={0 if nth_generation == 0 else 1}, extra_body={extra_body if extra_body else 'NOT_GIVEN'}")

		# Taiji always returns SSE (text/event-stream), so force streaming mode
		use_streaming = streaming or bool(config.get('taiji_wsid'))

		if use_streaming:
			stream = client.chat.completions.create(
				model=model,
				messages=messages,
				stream=True,
				max_tokens=max_tokens,
				temperature=0 if nth_generation == 0 else 1,
				timeout=180,
				extra_body=extra_body_arg
			)

			response = ""
			for chunk in stream:
				# Skip non-standard chunks (e.g. Taiji "thinking" plugin events) where choices is None
				if not chunk.choices:
					continue
				try:
					if chunk.choices[0].delta.content is not None:
						response += chunk.choices[0].delta.content
				except Exception as chunk_err:
					if len(response) == 0:
						logger.error(f"[get_response] Streaming failed on first chunk: {type(chunk_err).__name__}: {chunk_err}")
						logger.error(f"[get_response] Raw chunk: {chunk}")
						return None

					if len(chunk.choices) == 0 and response.strip()[-1] == '}':
						break
		else:
			completion = client.chat.completions.create(
				model=model,
				messages=messages,
				max_tokens=max_tokens,
				temperature=0 if nth_generation == 0 else 1,
				timeout=180,
				extra_body=extra_body_arg
			)
			response = completion.choices[0].message.content

		logger.info(f"[get_response] Success! Response length={len(response)} chars, preview: {response[:200]}...")
		return response

	except Exception as e:
		import traceback
		logger.error(f'Prompt: {str(messages)[:500]}')
		logger.error(f"Error in get_response: {str(e)}")

		try:
			if 'response' in dir() and response is not None:
				if hasattr(response, 'text'):
					logger.error(f"Response: {response.text}")
				else:
					logger.error(f"Response: {response}")
		except Exception as e2:
			logger.error(f"Could not print response: {e2}")
		
		logger.error(f"Number of input tokens: {num_tokens_from_string(messages[0]['content'])}")

		traceback.print_exc()
		return None
	
def lang_detect(text):
	import re
	def count_chinese_characters(text):
		chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
		return len(chinese_chars)
			
	if count_chinese_characters(text) > len(text) * 0.05:
		lang = 'zh'
	else:
		lang = 'en'
	return lang
	

def remove_inner_thoughts(dialogue: str) -> str:
	cleaned_dialogue = re.sub(r'\[.*?\]', '', dialogue)

	cleaned_dialogue = '\n'.join(line.strip() for line in cleaned_dialogue.split('\n'))
	
	cleaned_dialogue = re.sub(r'\n+', '\n', cleaned_dialogue)
	
	return cleaned_dialogue.strip()

def add_speaker_name(dialogue: str, speaker: str) -> str:
	# Check if the dialogue already contains a speaker prefix at the beginning of any line
	if any(line.strip().startswith(f"{speaker}:") or line.strip().startswith(f"{speaker}：") for line in dialogue.split('\n')):
		return dialogue
	
	# Add the speaker name at the beginning
	return f"{speaker}: {dialogue}"


def load_json(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data

def get_character_prompt(book_name, character, character_profile, background, scenario, motivation, thoughtless=False, other_character_profiles=None, exclude_plot_summary=False, fixed_template=False, add_output_example=False, add_rag=False):

	if thoughtless:
		output_format = "你的输出应包含说话内容和可见动作。可见动作请放在 `()` 中，例如 `(轻轻点头)`。除非原始场景本身包含英文，否则请使用中文。"
	else:
		output_format = "你的输出应包含内心想法、说话内容和可见动作。内心想法请放在 `[]` 中，例如 `[我必须稳住。]`；可见动作请放在 `()` 中，例如 `(攥紧袖口)`。如果是发言，请先写内心想法，再写说话和动作。除非原始场景本身包含英文，否则请使用中文。"

		if add_output_example:
			output_format = "你的输出应包含内心想法、说话内容和可见动作。格式示例：`[我心里发慌，但不能让人看出来。] “先别急，我们再看看。” (压低声音，努力保持镇定)`。除非原始场景本身包含英文，否则请使用中文。"

	if other_character_profiles:
		assert isinstance(other_character_profiles, Dict)
		other_character_profiles_str = ''
		for other_character, profile in other_character_profiles.items():
			if other_character != character:
				other_character_profiles_str += f"{other_character}：{profile}\n\n"
	else:
		other_character_profiles_str = ''

	if motivation:
		motivation_block = f"===当前心理与动机===\n{motivation}\n\n"
	else:
		motivation_block = ""

	if other_character_profiles_str:
		other_characters_block = f"===其他角色信息===\n{other_character_profiles_str}"
	else:
		other_characters_block = ""

	system_prompt = f"你现在要扮演《{book_name}》中的角色“{character}”。请严格按照该角色的身份、经历、性格、关系和当前处境来思考与回应。\n\n"
	system_prompt += f"===角色档案===\n{character_profile}\n\n"

	if not exclude_plot_summary and background:
		system_prompt += f"===相关剧情摘要===\n{background}\n\n"

	system_prompt += f"===当前场景===\n{scenario}\n\n"

	if other_characters_block:
		system_prompt += other_characters_block + "\n"

	system_prompt += motivation_block

	if add_rag:
		system_prompt += "===补充背景信息===\n{retrieved_knowledge}\n\n"

	system_prompt += "===要求===\n"
	system_prompt += "1. 你必须始终以该角色的身份、立场和认知范围作答，不要跳出角色解释设定。\n"
	system_prompt += "2. 回应必须符合中文小说语境，语气、措辞和行为要贴近人物与时代背景。\n"
	system_prompt += "3. 不要替其他主要角色发言，也不要代替“环境”描述全局变化，除非你的角色正在主动观察并表达。\n"
	system_prompt += f"4. {output_format}\n"
	system_prompt += "5. 不要无故输出英文 thought、英文动作描述或英文舞台说明。\n\n"

	return system_prompt

def get_environment_prompt(major_characters, scenario):
	ENVIRONMENT = "环境"
	major_characters = [c for c in major_characters if c != ENVIRONMENT]

	model_roles = [
		"环境模拟器",
		"世界模型",
		"场景模拟器",
		"叙事环境模型"
	]

	prompt = f"""你现在是角色扮演任务中的{random.choice(model_roles)}。你的职责是根据人物之间的互动、对话和动作，给出“环境”的反馈，也就是描述外部世界随之发生的变化。这包括：
   - 场景本身的物理变化
   - 背景人物、围观者或群众的反应
   - 声音、天气、光线、气氛等环境变化
   - 其他与场景推进相关的客观环境信息

你的描述要生动、简洁，并帮助建立场景氛围，但不要替主要角色（包括 {major_characters}）决定他们的具体台词或动作。

注意事项：
- 你可以描写次要人物或群体的反应，但不要代替主要角色发言。
- 输出应简洁有力，通常 1 到 3 句即可。
- 你的输出应与场景的时代、语气、文化背景和语言保持一致；中文场景请使用中文。
- 不要无故输出英文环境描述。

===当前场景如下===
{scenario}"""

	return prompt

def get_nsp_prompt(all_characters, scenario):
	ENVIRONMENT = "环境"

	prompt = f"""你的任务是为角色扮演场景预测“下一个发言者”。

你需要根据此前的互动，判断接下来最可能行动或说话的是谁。候选项只能从这个列表里选择：{all_characters}。其中“{ENVIRONMENT}”表示环境反馈，而不是具体人物。

输出规则：
- 如果能判断出下一位行动者，就只输出对应名字
- 如果无法判断，就输出 "random"
- 如果你认为这段场景已经自然结束，就输出 "<END CHAT>"

===当前场景如下===
{scenario}"""
	
	return prompt


from typing import Dict

def print_conversation_to_file(conversation_data: Dict, file_path: str):
	"""
	Write the scenario, actor prompt, user prompt, and the formatted conversation to a file.
	:param conversation_data: The dictionary containing scene details, actor prompt, user prompt, and conversation entries.
	:param file_path: The path to the file where the output will be written.
	"""
	# Extract components from the conversation data
	scene = conversation_data['scene']
	actor_prompt = conversation_data.get("actor_prompt", "N/A")
	user_prompt = conversation_data.get("user_prompt", "N/A")
	conversation = conversation_data["conversation"]

	with open(file_path, 'a', encoding='utf-8') as file:
		file.write("\n=== Scene Description ===\n")
		file.write(f"Scenario: {scene['scenario']}\n")
		
		file.write("\n=== Actor Prompt ===\n")
		file.write(f"{actor_prompt}\n")
		
		file.write("\n=== User Prompt ===\n")
		file.write(f"{user_prompt}\n")
		
		file.write("\n=== Conversation ===\n")
		for turn in conversation:
			from_ = turn["from"]
			file.write(f"\n=== {from_} ===\n")
			message = turn["message"]
			file.write(f"{message}\n\n")

	return 


def extract_json(text, **kwargs):
	def _fix_json(json_response):
		
		prompt = f'''I will provide you with a JSON string that contains errors, making it unparseable by `json.loads`. The most common issue is the presence of unescaped double quotes inside strings. Your task is to output the corrected JSON string. The JSON string to be corrected is:
{json_response}
'''

		response = get_response(model=kwargs['model'], messages=[{"role": "user", "content": prompt}])

		logger.info(f'fixed json: {response}')	

		return response
	
	def _fix_json_truncated(json_response):
		
		prompt = f'''I will provide you with a JSON string that contains errors, making it unparseable by `json.loads`. Your task is to correct these errors and output a valid JSON string. Please consider the following common issues and apply the appropriate fixes:

1. Unescaped double quotes inside strings: Escape these quotes properly.
2. Truncated JSON: If the JSON appears to be truncated, especially in cases where it contains multiple "plots" and each "plot" contains multiple "conversations", please:
   a) Identify the last complete structure (plot or conversation).
   b) Remove any incomplete trailing content.
   c) Add the appropriate closing brackets or braces (e.g., "}}" or "]") to ensure valid JSON structure.
3. Other syntax errors: Correct any other JSON syntax errors you may encounter.

Please analyze and correct the following JSON string:

{json_response}

Output only the corrected JSON string, without any additional explanations or comments.'''

		response = get_response(model="claude-3-5-sonnet-20240620", messages=[{"role": "user", "content": prompt}])

		logger.info(f'fixed json: {response}')	

		return response

	def _extract_json(text):
		# Use regular expressions to find all content within curly braces
		orig_text = text

		text = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', lambda m: m.group().replace('\n', r'\\n'), text) 
		
		#json_objects = re.findall(r'(\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)

		def parse_json_safely(text):
			try:
				result = json.loads(text)
				return result
			except json.JSONDecodeError:
				results = []
				start = 0
				while start < len(text):
					try:
						obj, end = json.JSONDecoder().raw_decode(text[start:])
						results.append(obj)
						start += end
					except json.JSONDecodeError:
						start += 1
				
				if results:
					longest_json = max(results, key=lambda x: len(json.dumps(x)))
					return longest_json
				else:
					return None
		
		extracted_json = parse_json_safely(text)
		
		if extracted_json:
			return extracted_json
		else:
			logger.error('Error parsing response: ', orig_text)
			return None

	# an inserted workflow for post processing in restore_from_cache
	if kwargs.get('post_fix_truncated_json_', False):
		text = _fix_json_truncated(text)

		res = _extract_json(text)

		return res 
	

	res = _extract_json(text)

	if res:
		return res
	else:
		if kwargs.get('fix_truncated_json', False):
			return _extract_json(_fix_json_truncated(text))
		else:
			return _extract_json(_fix_json(text))


def get_response_json(post_processing_funcs=[extract_json], **kwargs):
    """
    Get and process a response from an LLM with retries and error handling.
    
    This function handles:
    1. Getting responses from the LLM with retries
    2. Handling copyright warnings by adjusting the prompt
    3. Processing responses through a pipeline of post-processing functions
    4. Fallback handling for parsing failures
    
    Args:
        post_processing_funcs (list): List of functions to process the LLM response, defaults to [extract_json]
        **kwargs: Additional arguments passed to get_response(), including:
            - messages: List of message dicts for the LLM
            - model: Name of LLM model to use
            - max_retry: Max number of retry attempts (default 5)
            
    Returns:
        dict: Processed JSON response from the LLM, or error dict if parsing fails
    """
    nth_generation = 0  # Track number of retry attempts
    secondary_response = None  # Store backup response for parsing failures
    none_count = 0  # Track consecutive None responses

    while True:
        logger.info(f'{nth_generation}th generation')
        response = get_response(**kwargs, nth_generation=nth_generation)
        logger.info(f'response by LLM: {response}')

        if response is None:
            none_count += 1
            if none_count >= 3:
                logger.error(f'[get_response_json] Got {none_count} consecutive None responses, aborting.')
                return None
            nth_generation += 1
            continue
        none_count = 0

        # Reset to single message if we previously added copyright handling messages
        if len(kwargs['messages']) > 1:
            kwargs['messages'] = kwargs['messages'][:1]

        # Check for copyright warning in short responses
        words = response.split(' ')
        if len(words) < 100 and 'reproduce' in response and 'copyright' in response and len(kwargs['messages']) == 1:
            # Add messages to handle copyright warning and request appropriate summary
            warning = "I will not reproduce any copyrighted material. However, I'd be happy to provide a summary of the key plot points and character interactions from the given book excerpt, while being careful not to include any lengthy quotes or passages. Please let me know if you would like me to provide that type of summary."
            kwargs['messages'].append({"role": "assistant", "content": warning})
            kwargs['messages'].append({"role": "user", "content": "Yes, please provide that type of summary, but remember to follow my requirements."})
            
            nth_generation += 1
            continue

        # Run response through post-processing pipeline
        for i, post_processing_func in enumerate(post_processing_funcs):
            if response is None:
                break
            
            prev_response = response
            response = post_processing_func(response, **kwargs)

            # Special handling for parse_response failures
            if post_processing_func.__name__ == 'parse_response' and response == False:
                orig_response = get_response(**kwargs, nth_generation=nth_generation)

                # Store longest response as backup
                if secondary_response:
                    if len(orig_response) > len(secondary_response):
                        secondary_response = orig_response
                else:
                    secondary_response = orig_response

                logger.info(f'orig_response: {orig_response}\nNum Tokens: {num_tokens_from_string(orig_response)}')

        json_response = response

        # Break if we got a valid response, otherwise retry
        if json_response:
            break
        else:
            nth_generation += 1
            if nth_generation > kwargs.get('max_retry', 5):
                # Return error response with backup data if parse_response failed
                if 'parse_response' in [f.__name__ for f in post_processing_funcs]:
                    return {"fail_to_parse_response": secondary_response}

    return json_response

def print_json(data):
	logger.info(json.dumps(data, ensure_ascii=False, indent=2))

def save_json(data: List[Dict], file_path: str):
	with open(file_path, "w", encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(file_path: str) -> List[Dict]:
	with open(file_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data

	
if __name__ == '__main__':
	messages = [{"role": "system", "content": "Hello, how are you? Hello, how are you? Hello, how are you?"}]
	model = 'gpt-4o'

	print(get_response(model, messages))
		
