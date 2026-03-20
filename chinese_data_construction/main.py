# Standard library imports
import argparse
import json
import os
import re
import traceback
from datetime import datetime
from collections import Counter
from typing import List, Optional, Tuple
# Third-party imports
try:
    import jsonlines
except ImportError:
    jsonlines = None
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
import difflib

RUN_TIMESTAMP_ENV = 'CHINESE_DC_RUN_TIMESTAMP'
RUN_OUTPUT_DIR_ENV = 'CHINESE_DC_RUN_OUTPUT_DIR'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Construct CoSER-style dataset from prepared JSONL book records'
    )
    parser.add_argument('--input', type=str, required=True,
                      help='Input JSONL path. Each record must already contain title/author/content.')
    parser.add_argument('--output_dir', type=str, default='data',
                      help='Output directory path (default: data')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of parallel workers (default: 1)')
    parser.add_argument('--model', type=str, default="gpt-4o",
                      help='Model to use for data construction (default: gpt-4o)')
    parser.add_argument('--candidate_model', type=str, default="gpt-4o",
                      help='Another candidate model to use for data construction when the main model fails (default: gpt-4o)')
    parser.add_argument('--regenerate', action='store_true',
                      help='Force regenerate data even if results already exist (default: False)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def prepare_run_output_dir(args):
    """Create a per-run output directory and reuse it across worker processes."""
    base_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(base_output_dir, exist_ok=True)

    run_timestamp = os.environ.get(RUN_TIMESTAMP_ENV)
    if not run_timestamp:
        run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.environ[RUN_TIMESTAMP_ENV] = run_timestamp

    run_output_dir = os.environ.get(RUN_OUTPUT_DIR_ENV)
    if not run_output_dir:
        run_output_dir = os.path.join(base_output_dir, run_timestamp)
        os.environ[RUN_OUTPUT_DIR_ENV] = run_output_dir

    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(os.path.join(run_output_dir, 'extracted'), exist_ok=True)
    os.makedirs(os.path.join(run_output_dir, 'final'), exist_ok=True)

    args.base_output_dir = base_output_dir
    args.run_timestamp = run_timestamp
    args.output_dir = run_output_dir
    return args

args = prepare_run_output_dir(parse_args())

# Local imports
from utils import config, cached, get_response, setup_logger, get_response_json, print_json, encode, decode

# Setup logger
logger = setup_logger(__name__, os.path.join(args.output_dir, 'main.log'))

def find_index(lst, key):
    """
    Find the index of a key in a list, returning -1 if not found.

    Args:
        lst: The list to search in
        key: The key to search for

    Returns:
        int: The index of the key if found, -1 if not found
    """
    try:
        return lst.index(key)
    except ValueError:
        return -1


MATCH_REPLACEMENTS = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "—": "-",
    "–": "-",
    "－": "-",
    "…": "...",
    "·": ".",
    "．": ".",
    "。": "。",
    "！": "！",
    "？": "？",
    "\u3000": " ",
}
COPYRIGHT_KEYWORDS_EN = ["rights", "reserved", "reproduced", "copyright", "reproduce", "permission"]
COPYRIGHT_KEYWORDS_ZH = [
    "版权所有",
    "版权归",
    "请勿用于商业",
    "请支持正版",
    "转载",
    "互联网",
    "电子书",
    "论坛",
    "公众号",
    "下载",
    "删除",
]


def normalize_for_match(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text)
    for old, new in MATCH_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def build_normalized_text_map(text: str):
    normalized_chars = []
    index_map = []
    for i, char in enumerate(text):
        replacement = MATCH_REPLACEMENTS.get(char, char)
        if char.isspace():
            continue
        for normalized_char in replacement:
            if normalized_char.isspace():
                continue
            normalized_chars.append(normalized_char)
            index_map.append(i)
    return "".join(normalized_chars), index_map


def locate_text_span(text: str, target: Optional[str]):
    if not target:
        return None
    exact_index = text.find(target)
    if exact_index != -1:
        return exact_index, exact_index + len(target)

    normalized_target = normalize_for_match(target)
    if not normalized_target:
        return None
    normalized_text, index_map = build_normalized_text_map(text)
    normalized_index = normalized_text.find(normalized_target)
    if normalized_index == -1:
        return None
    start = index_map[normalized_index]
    end = index_map[normalized_index + len(normalized_target) - 1] + 1
    return start, end


def extract_original_text_span(chunk: str, first_sentence: Optional[str], last_sentence: Optional[str]) -> str:
    if not first_sentence or not last_sentence:
        return ""
    first_span = locate_text_span(chunk, first_sentence)
    last_span = locate_text_span(chunk, last_sentence)
    if not first_span or not last_span:
        return ""

    start = first_span[0]
    end = last_span[1]
    if end < start:
        next_last = chunk.find(last_sentence, start)
        if next_last != -1:
            end = next_last + len(last_sentence)
        else:
            return ""
    return chunk[start:end]


def build_remaining_chunk(chunk: str, next_chunk_start: Optional[str]) -> str:
    if not next_chunk_start:
        return ""
    span = locate_text_span(chunk, next_chunk_start)
    if not span:
        return ""
    remaining_chunk = chunk[span[0]:]
    remaining_chunk = remaining_chunk[int(len(remaining_chunk) * 0.2):]
    newline_index = find_index(remaining_chunk, '\n')
    if newline_index != -1:
        remaining_chunk = remaining_chunk[newline_index + 1:]
    return remaining_chunk.strip()


def split_text_by_tokens(text: str, chunk_size: int) -> List[str]:
    tokens = encode(text)
    results = []
    start_index = 0

    while start_index < len(tokens):
        if len(tokens) - start_index <= chunk_size:
            results.append(decode(tokens[start_index:]))
            break
        chunk_tokens = tokens[start_index:start_index + chunk_size]
        results.append(decode(chunk_tokens))
        start_index += len(chunk_tokens)
    return [chunk for chunk in results if chunk]


def split_text_into_sentences(text: str) -> List[str]:
    sentences = []
    current = []
    idx = 0
    closing_chars = '”’」』】）)"\''
    sentence_end_chars = '.!?。！？…'

    while idx < len(text):
        char = text[idx]
        current.append(char)

        if char == '\n':
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
            idx += 1
            continue

        if char in sentence_end_chars:
            while idx + 1 < len(text) and text[idx + 1] in closing_chars:
                idx += 1
                current.append(text[idx])
            sentence = ''.join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
        idx += 1

    tail = ''.join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_text_by_paragraphs(text: str, chunk_size: int) -> List[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r'\n{2,}', text) if paragraph.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [line.strip() for line in text.split('\n') if line.strip()]
    if not paragraphs:
        return split_text_by_tokens(text, chunk_size)

    results = []
    current_chunk = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = len(encode(paragraph))
        if paragraph_tokens >= chunk_size:
            if current_chunk:
                results.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            results.extend(split_text_by_tokens(paragraph, chunk_size))
            continue

        if current_chunk and current_tokens + paragraph_tokens > chunk_size:
            results.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_tokens = 0

        current_chunk.append(paragraph)
        current_tokens += paragraph_tokens

    if current_chunk:
        results.append('\n\n'.join(current_chunk))
    return [chunk for chunk in results if chunk]


def is_probable_copyright_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lower_line = stripped.lower()
    english_hits = sum(keyword in lower_line for keyword in COPYRIGHT_KEYWORDS_EN)
    chinese_hits = sum(keyword in stripped for keyword in COPYRIGHT_KEYWORDS_ZH)
    return len(stripped) < 160 and (english_hits >= 2 or chinese_hits >= 2)


def read_jsonl_records(file_path: str):
    if jsonlines is not None:
        with jsonlines.open(file_path, mode='r') as reader:
            return list(reader)
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_input_books(books_data, input_path: str):
    """
    Validate that the pipeline input is already a normalized JSONL dataset.

    This pipeline now consumes prepared records directly instead of invoking
    the legacy txt preprocessing stage.
    """
    required_fields = ("title", "author", "content")
    invalid_examples = []

    for idx, book in enumerate(books_data):
        if not isinstance(book, dict):
            invalid_examples.append(
                f"line {idx + 1}: expected JSON object, got {type(book).__name__}"
            )
            if len(invalid_examples) >= 5:
                break
            continue

        missing_fields = [field for field in required_fields if field not in book]
        empty_fields = [
            field for field in required_fields
            if field in book and not str(book[field]).strip()
        ]

        if missing_fields or empty_fields:
            problems = []
            if missing_fields:
                problems.append(f"missing {', '.join(missing_fields)}")
            if empty_fields:
                problems.append(f"empty {', '.join(empty_fields)}")
            invalid_examples.append(f"line {idx + 1}: {'; '.join(problems)}")
            if len(invalid_examples) >= 5:
                break

    if invalid_examples:
        details = "\n".join(f"  - {example}" for example in invalid_examples)
        raise ValueError(
            "Input JSONL is not compatible with chinese_data_construction.\n"
            "Expected each record to already include non-empty title, author, and content fields.\n"
            "The pipeline no longer runs preprocess.py as part of data construction.\n"
            f"Input: {input_path}\n"
            f"Examples:\n{details}"
        )

@cached
def create_chunk_generator(book, chunk_size):
    """
    Generates chunks of text from a book while respecting token limits and chapter boundaries.

    Args:
        book (dict): A dictionary containing book information with 'content' and other fields
        chunk_size (int): Roughly the number of tokens per chunk

    Returns:
        list: A list of text chunks from the book, where each chunk is:
            - Limited to chunk_size if no chapters are detected
            - Between chunk_size/2 and 2*chunk_size if chapters are detected
            - Cleaned of copyright notices in the first chunk
            - Cleaned of excessive tabs if present

    The function handles books in two ways:
    1. For books without chapter markers: Splits into fixed-size chunks of chunk_size
    2. For books with chapters: Attempts to keep chapters together while staying within token limits
    """
    # Check and clean excessive tabs that may interfere with text processing
    def has_excessive_tabs(content, threshold=0.05):
        tab_count = content.count('\t')
        return (tab_count / len(content)) > threshold
    
    if has_excessive_tabs(book['content']):
        book['content'] = book['content'].replace('\t', '')

    from split import split_book
    chapters = split_book(book)
    results = []

    if not chapters:
        results = split_text_by_paragraphs(book['content'], chunk_size)
    else:
        current_chunk = []
        current_tokens = 0
        for chapter in chapters:
            current_chunk.append(chapter['content'])
            current_tokens += len(encode(chapter['content']))
            if current_tokens >= chunk_size // 2:
                if current_tokens <= 2 * chunk_size:
                    results.append(''.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                else:
                    chunk_text = ''.join(current_chunk)
                    results.extend(split_text_by_paragraphs(chunk_text, chunk_size))
                    current_chunk = []
                    current_tokens = 0
        if current_chunk:
            results.append(''.join(current_chunk))

    if not results:
        results = [book['content']]

    lines = results[0].split('\n')
    filtered_lines = []
    for line in lines:
        if is_probable_copyright_line(line):
            continue
        filtered_lines.append(line)
    results[0] = '\n'.join(filtered_lines).strip()

    return results


def ngram_jaccard_similarity(text1, text2, n=3):
    """Calculate the Jaccard similarity between two texts using n-grams.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare 
        n (int, optional): Size of n-grams. Defaults to 3.
    
    Returns:
        float: Jaccard similarity score between 0 and 1, where 1 means identical texts
              and 0 means completely different texts.
    """
    def ngrams(tokens, n):
        """Generate n-grams from a sequence of tokens.
        
        Args:
            tokens (list): List of tokens
            n (int): Size of n-grams
        Returns:
            list: List of n-gram tuples
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def jaccard_similarity(set1, set2):
        """Calculate Jaccard similarity between two sets.
        
        Args:
            set1 (set): First set
            set2 (set): Second set
        Returns:
            float: Jaccard similarity score
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union != 0 else 0

    # Tokenize the input texts into sequences of tokens
    tokens1 = encode(text1)
    tokens2 = encode(text2)
    
    # Generate sets of n-grams from the token sequences
    ngrams1 = set(ngrams(tokens1, n))
    ngrams2 = set(ngrams(tokens2, n))
    
    # Calculate and return the Jaccard similarity between the n-gram sets
    return jaccard_similarity(ngrams1, ngrams2)

@cached
def find_best_match_passage(candidates, target, n=3, threshold=0.3):
    """Find the best matching passage from a list of candidates compared to a target text. These texts are generally LLM-synthesized summaries. Hence, we focus on their semantic similarity.

    Uses n-gram Jaccard similarity to compare texts and find the closest match.
    
    Args:
        candidates (list): List of candidate passages to search through
        target (str or dict): Target text to match against
        n (int, optional): Size of n-grams to use for comparison. Defaults to 3.
        threshold (float, optional): Minimum similarity score required to consider a match.
                                   Defaults to 0.3.
    
    Returns:
        int: Index of best matching passage if score >= threshold, -1 if no good match found
    """
    best_match = None  # Index of current best matching passage
    best_score = 0     # Highest similarity score found so far

    # Handle case where inputs are dictionaries by converting to strings
    if isinstance(candidates, list) and isinstance(target, dict) and isinstance(candidates[0], dict):
        target = str(target)
        candidates = [str(c) for c in candidates]

    # Compare target against each candidate passage
    for i, candidate in enumerate(candidates):
        score = ngram_jaccard_similarity(target, candidate, n)
        if score >= best_score:
            best_score = score
            best_match = i
    
    # Return best match if it meets threshold, otherwise return -1
    if best_score >= threshold:
        logger.debug(f"Best match: \nInput: {target}\nOutput: {candidates[best_match]}\nScore: {best_score}")
        return best_match
    else:
        return -1


@cached
def find_best_match_sentence(chunk, target, threshold=0.6):
    """Find the best matching sentence from a chunk of text or list of sentences compared to a target sentence. These sentences are generally exact sentences from the book, so we focus on their string similarity.
    
    Uses SequenceMatcher to calculate string similarity ratios between sentences to find the closest match.
    
    Args:
        chunk (str or list): Input text chunk or list of sentences to search through
        target (str): Target sentence to match against
        threshold (float, optional): Minimum similarity score required to consider a match.
                                   Defaults to 0.6.
    
    Returns:
        str or None: Best matching sentence if score >= threshold, None if no good match found
                    or if target is None/invalid
    """
    # Return None for invalid target inputs
    if target == 'None' or target is None:
        return None

    if isinstance(chunk, str):
        sentences = split_text_into_sentences(chunk)
    else:
        assert isinstance(chunk, list)
        sentences = chunk

    best_match = None
    best_score = 0

    normalized_target = normalize_for_match(target)
    for sentence in sentences:
        normalized_sentence = normalize_for_match(sentence)
        if not normalized_sentence:
            continue
        score = difflib.SequenceMatcher(None, normalized_target, normalized_sentence).ratio()
        if score > best_score:
            best_score = score
            best_match = sentence
    logger.debug(f"Best match: \nInput: {target}\nOutput: {best_match}\nScore: {best_score}")

    if best_score >= threshold:
        return best_match
    return None
def extract_from_chunk(book, i_c, chunk, truncated_plots=None):
    """
    Extract and process plot information from a chunk of book text.
    
    This function analyzes a chunk of text to identify chapter beginnings, plots, conversations,
    and other narrative elements. It uses an LLM to generate structured information about the text.

    Args:
        book (dict): Dictionary containing book metadata including title and author
        i_c (int): Chunk index
        chunk (str): Text content of the current chunk to analyze
        truncated_plots (list, optional): List of incomplete plots from previous chunk that need to be finished

    Returns:
        tuple: Contains:
            - chapter_beginnings (list): List of identified chapter starts
            - plots (list): Extracted plot information including summaries, characters, conversations
            - remaining_chunk (str): Unused portion of chunk to process in next iteration
            
    The function generates a detailed prompt for the LLM that requests:
    1. Chapter beginning identification
    2. Plot extraction and analysis
    3. Conversation reconstruction
    4. Character motivation analysis
    5. Next chunk starting point determination
    """
    logger.info(f"Extracting plots from chunk for book: {book['title']}")

    # Create deep copy of truncated plots and remove text field to avoid redundancy
    import copy
    if truncated_plots:
        truncated_plots = copy.deepcopy(truncated_plots)
        for plot in truncated_plots:
            plot.pop('text')
    
    # Construct the prompt for the LLM
    prompt = f"""
基于给定的小说正文片段，完成以下任务：

1. 如果该片段中出现了新的章节开头，请识别出来，并给出该章节的起始句。
2. 识别该片段中的重要剧情。对每段剧情，标出它在本片段中的起始句和结束句，判断它所属的章节标题，并设置 "state"：
   - 如果该剧情在当前片段中尚未结束，设为 "truncated"
   - 否则设为 "finished"
   你还会收到上一片段中未结束的剧情；对于这些剧情，你**必须**结合当前片段继续补全对应的对话内容，但要保持原有的 **scenario** 不变。
3. 总结每段重要剧情。对每段剧情生成摘要，给出 1 到 100 的重要度评分，并列出关键角色以及他们在该剧情中的身份、想法与行为。
4. 为每段剧情抽取或构造对话。先给出这段对话开始时的场景说明（scenario）和主题（topic），再列出关键角色及其当前动机（motivation），最后给出对话内容。要求如下：
   i) 对话必须忠于原文剧情与人物设定，尽可能贴近原文已有对话，不要引入脱离上下文的剧情。
   ii) 对话要完整覆盖关键交流与信息，每段对话至少包含 20 个 utterance，鼓励丰富的对话内容；如果原文对话不足，则舍弃这段剧情。
   iii) 每个 utterance 由“内心想法 + 说话内容 + 可见动作”组成。内心想法必须放在 `[]` 中，例如 `[我心里发紧，但不能露怯。]`；可见动作必须放在 `()` 中，例如 `(沉默片刻)` 或 `(抬头看向对方)`。除“环境”外，每个 utterance 都应先写内心想法，再写说话和动作。
   iv) [重要] 生成内心想法时，要从角色视角出发，分析其言行背后的心理活动。这些想法应体现角色的背景、性格、价值观、人际关系、动机与目标，并写成完整短句，不要只写形容词或副词。
   v) 你还需要把场景、氛围、突发事件等环境信息写成特殊的 utterance，并将其 `character` 字段固定写为 `"环境"`。这类环境 utterance 不应包含角色的主动想法、观察或动作。
   vi) 对话内容必须与片段语言保持一致。当前片段是中文时，`scenario`、`topic`、`motivation`、`dialogues.message`、方括号内想法、圆括号内动作、以及环境描述都必须使用中文。除非原文本身就包含英文，否则不要输出英文单词或英文句子。
5. 识别下一片段的最佳起始点。如果最后一条剧情仍为 truncated，则将 next_chunk_start 设为 None；否则将其设为最后一条剧情的第一句。

===输出格式===
请严格按照以下 JSON 格式输出：
{{
    "chapter_beginnings": [
        {{
            "beginning_sentence": "该章节在正文中的起始句，通常就是章节标题原文。"
        }}
    ],
    "plots": [
        // 先补全上一片段遗留的 truncated plots（如果有）
        {{
            ...
        }}, 
        // 再输出当前片段中新识别出的 plots
        {{
            "chapter_title": "该剧情所属章节标题；如果无法确定则输出 None。",
            "first_sentence": "该剧情在当前 **chunk** 中的第一句原文。",
            "last_sentence": "该剧情在当前 **chunk** 中的最后一句原文；如果当前 chunk 内剧情被截断，就填当前 chunk 中该剧情实际出现的最后一句。",
            "prominence": "该剧情的重要程度，1 到 100。",
            "summary": "该剧情的简洁摘要，只做总结，不要扩写无关内容。",
            "key_characters": [
                {{
                    "name": "角色姓名",
                    "description": "该剧情开始前，这个角色的简要身份描述（约 20 字）。",
                    "experience": "该角色在这段剧情中的作用、想法、行为，以及与本剧情相关的重要变化（约 30 字）。",
                }}
            ],
            "conversation": [{{
                "scenario": "这段对话开始时的场景说明，要尽量交代清楚上下文，但不要提前泄露后续对话本身会呈现的信息。",
                "topic": "对话主题（尽量简洁，约 10 个字）", 
                "key_characters": [
                    {{
                        "name": "角色姓名",
                        "motivation": "该角色在对话开始前的想法，包含其态度、情绪、动机、目标、想传达的信息或想讨论的话题。必须使用中文。",
                    }}
                ],
                "dialogues": [
                    {{
                        "character": "角色姓名，或者固定写为“环境”",
                        "message": "utterance 内容。若 character 不是“环境”，格式应类似：[我强压着怒气。] “你先听我说。” (攥紧袖口)；若 character 是“环境”，则只描述环境变化，例如：(夜风穿过树梢，火光忽明忽暗)。除非原文本身有英文，否则这里必须使用中文。"
                    }}
                ]
            }}],
            "state": "finished 或 truncated"
        }}
    ],
    "next_chunk_start": "下一片段建议起始句。"
}}

===补充要求===
1. 必须严格遵守上述 JSON 格式。
2. [重要] 所有字符串中的双引号都必须正确转义，尤其是在抽取原文时。
3. 输出中尽量使用角色的正式全名，不要保留“少爷”“夫人”“舅舅”这类称呼作为主名字，除非正文里确实无法确定姓名。
4. 必须忠于原作内容，不要引入脱离上下文的剧情。若剧情中存在原文对话，优先保留原文对话风格；若原文对话不足，可补充与剧情一致的自然中文对话。
5. 禁止为了格式示例而输出英文，也不要把环境角色写成 "Environment"。

===输入===

==书名==
{book['title']}

==作者==
{book['author']}

==小说正文片段== 
{chunk}

==上一片段中尚未结束、需要继续补全的剧情==
{json.dumps(truncated_plots, ensure_ascii=False, indent=2) if truncated_plots else "None"}
"""
    
    # Example format for character utterances in conversations
    # "[My father's words fill me with awe, but I still feel uneasy.] 
    # (Nods seriously, but with a slight frown remaining) 
    # I understand, Father. Responsibility is important. But… is killing really necessary? 
    # (A flash of compassion in his eyes)
    # If someone has done something wrong, can't we give them a chance to make amends?"

    logger.debug(prompt)

    def parse_response(response, chunk, book, **kwargs):
        """
        Parse and validate the LLM response, extracting structured plot information.
        
        Args:
            response: Raw LLM response to parse
            chunk: Original text chunk for reference
            book: Book metadata
            **kwargs: Additional keyword arguments
            
        Returns:
            tuple or bool: (chapter_beginnings, plots, remaining_chunk) if successful, False if failed
        """
        if not response:
            return False
        
        try:
            # Handle different response formats
            # Sometimes response is just a single plot dict
            if (isinstance(response, dict) and 'first_sentence' in response):
                response = {
                    'chapter_beginnings': [],
                    'plots': [response],
                    'next_chunk_start': None
                }
            # Sometimes response is a list of plots
            elif isinstance(response, list):
                # Filter out non-dict items and wrap into standard format
                plot_items = [item for item in response if isinstance(item, dict)]
                response = {
                    'chapter_beginnings': [],
                    'plots': plot_items,
                    'next_chunk_start': None
                }
            # If response is somehow not a dict at this point, bail out
            elif not isinstance(response, dict):
                logger.warning(f"Unexpected response type {type(response)}, skipping")
                return False

            chapter_beginnings = response.get('chapter_beginnings', [])

            plots = []

            if response.get('next_chunk_start'):
                response['next_chunk_start'] = find_best_match_sentence(chunk, response['next_chunk_start'])
                remaining_chunk = build_remaining_chunk(chunk, response['next_chunk_start'])
            else:
                remaining_chunk = ''

            # Process each plot from the response
            for unprocessed_plot in response.get('plots', []):

                chapter_title = unprocessed_plot.get('chapter_title')

                unprocessed_plot['first_sentence'] = find_best_match_sentence(chunk, unprocessed_plot.get('first_sentence'))
                unprocessed_plot['last_sentence'] = find_best_match_sentence(chunk, unprocessed_plot.get('last_sentence'))

                first_sentence, last_sentence = unprocessed_plot['first_sentence'], unprocessed_plot['last_sentence']

                original_text = extract_original_text_span(chunk, first_sentence, last_sentence)

                # Normalize key_characters
                if not isinstance(unprocessed_plot.get('key_characters'), list):
                    unprocessed_plot['key_characters'] = []
                for kc in unprocessed_plot['key_characters']:
                    if isinstance(kc, dict) and 'name' not in kc and 'character' in kc:
                        kc['name'] = kc.pop('character')

                # Normalize conversations
                normalize_plot_conversations(unprocessed_plot)

                # Create structured plot object
                plot = {
                    'text': original_text,
                    'summary': unprocessed_plot.get('summary', ''),
                    'prominence': unprocessed_plot.get('prominence', 50),
                    'key_characters': unprocessed_plot['key_characters'],
                    'chapter': chapter_title,
                    'conversation': unprocessed_plot['conversation'],
                    'state': unprocessed_plot.get('state', 'finished')
                }

                plots.append(plot)

            # Log processed response
            print_json(response)
            logger.info(json.dumps(response, ensure_ascii=False, indent=2))

            return chapter_beginnings, plots, remaining_chunk

        except Exception as e:
            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
            return False

    from utils import get_response_json, extract_json

    # Get and parse LLM response
    response = get_response_json([extract_json, parse_response], model=args.model, messages=[{"role": "user", "content": prompt}], book=book, chunk=chunk, fix_truncated_json=True)

    return response


def normalize_conversation(conversation):
    """标准化单个 conversation 对象，确保字段名和类型一致。"""
    # 如果 conversation 不是 dict，返回 None 标记为无效
    if not isinstance(conversation, dict):
        logger.warning(f"Invalid conversation type: {type(conversation)}, skipping")
        return None

    # 确保 scenario 存在
    if 'scenario' not in conversation:
        conversation['scenario'] = ''

    # 确保 topic 存在
    if 'topic' not in conversation:
        conversation['topic'] = ''

    # 标准化 key_characters：确保每个元素有 name 字段
    if 'key_characters' not in conversation or not isinstance(conversation['key_characters'], list):
        conversation['key_characters'] = []
    for kc in conversation['key_characters']:
        if isinstance(kc, dict) and 'name' not in kc and 'character' in kc:
            kc['name'] = kc.pop('character')

    # 标准化 dialogues 字段名
    if 'dialogues' not in conversation:
        if 'dialogue' in conversation:
            conversation['dialogues'] = conversation.pop('dialogue')
        else:
            conversation['dialogues'] = []

    # 确保 dialogues 是 list
    if not isinstance(conversation['dialogues'], list):
        conversation['dialogues'] = []

    # 标准化每个 utterance 的 character 字段
    for utt in conversation['dialogues']:
        if isinstance(utt, dict) and 'character' not in utt and 'name' in utt:
            utt['character'] = utt.pop('name')

    return conversation


def normalize_plot_conversations(plot):
    """标准化 plot 中的 conversation 字段，确保为 list[dict] 且每个元素字段完整。"""
    conv = plot.get('conversation')

    # None 或缺失 -> 空列表
    if conv is None:
        plot['conversation'] = []
        return

    # 单个 dict -> 包成列表
    if isinstance(conv, dict):
        conv = [conv]

    # 如果是 str（LLM 异常返回），丢弃并记录
    if isinstance(conv, str):
        logger.warning(f"conversation is a string, discarding: {conv[:200]}")
        plot['conversation'] = []
        return

    # 确保是 list
    if not isinstance(conv, list):
        logger.warning(f"conversation is unexpected type {type(conv)}, discarding")
        plot['conversation'] = []
        return

    # 逐个标准化，过滤无效项
    plot['conversation'] = [c for c in (normalize_conversation(c) for c in conv) if c is not None]


def extract(book, chunk_size=8192):
    """Process a book by splitting it into chunks and extracting structured information.

    This function processes a book by:
    1. Splitting the book text into chunks of specified size
    2. Extracting chapter beginnings, plots and conversations from each chunk
    3. Handling truncated plots that span multiple chunks by merging them
    4. Saving the extracted results to a JSON file

    Args:
        book (dict): Book data containing 'title', 'author', and 'content'
        chunk_size (int, optional): Roughly the number of tokens per chunk. Defaults to 8192.

    Returns:
        dict: Extracted results containing:
            - chapter_beginnings: List of chapter names (start locations)
            - plots: List of extracted plots with conversations
            - fail_to_parse_responses: List of chunks that failed parsing
    """
    # Set up save path and skip if already processed
    save_dir = f'{args.output_dir}/extracted'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{book["title"]}.json'
    if os.path.exists(save_path) and not args.regenerate:
        print(f"  [extract] Skipping (already exists): {save_path}")
        return 

    # Set up cache path
    from utils import set_cache_path
    set_cache_path(f'.cache/{book["title"]}.pkl')

    # Create generator to iterate through book chunks
    chunk_generator = create_chunk_generator(book, chunk_size)

    # Initialize results structure
    results = {
        'chapter_beginnings': [],
        'plots': [],
    }

    # Track state between chunks
    remaining_chunk = ''  # Text carried over from previous chunk
    truncated_plots = []  # Plots that continue into next chunk
    fail_to_parse_responses = []  # Track all failed parses across chunks

    # Process each chunk
    for i, chunk in enumerate(chunk_generator):

        print(f"  [extract] Processing chunk {i} ({len(encode(chunk))} tokens) for: {book['title']}")

        # Extract information from current chunk
        print(f"  [extract] Calling LLM for chunk {i}...")
        response = extract_from_chunk(book, i, (remaining_chunk or '') + chunk, truncated_plots)

        # Handle the response
        if response:
            if isinstance(response, tuple) and len(response) == 3:
                # Successful extraction
                chapter_beginnings, plots, remaining_chunk = response
            else:
                # Failed extraction
                chapter_beginnings, plots, remaining_chunk = [], [], ''
                if isinstance(response, dict) and 'fail_to_parse_response' in response:
                    fail_to_parse_responses.append(response['fail_to_parse_response'])
        else:
            # No response
            chapter_beginnings, plots, remaining_chunk = [], [], ''

        # Merge truncated plots from previous chunk with current plots
        for u_plot in truncated_plots:
            # Find matching plot in current chunk (by summary similarity)
            idx = find_best_match_passage([p['summary'] for p in plots], u_plot['summary'])

            if idx != -1:
                try:
                    # Found matching plot - merge them
                    plots[idx]['text'] = u_plot['text'] + plots[idx]['text']

                    # Ensure both sides have normalized conversations
                    normalize_plot_conversations(u_plot)
                    normalize_plot_conversations(plots[idx])

                    # Merge conversations
                    old_conversations = u_plot['conversation']
                    new_conversations = plots[idx]['conversation']
                    merged_conversations = []

                    # Check each previous conversation
                    for prev_conv in old_conversations:
                        idx_c = find_best_match_passage(
                            [s.get('scenario', '') for s in new_conversations],
                            prev_conv.get('scenario', '')
                        )

                        if idx_c != -1:
                            # Use new conversation if scenarios match
                            merged_conversations.append(new_conversations[idx_c])
                        else:
                            # Keep old conversation if no match
                            merged_conversations.append(prev_conv)

                    # Add any new conversations not already merged
                    merged_conversations += [ c for c in new_conversations if c not in merged_conversations ]
                    plots[idx]['conversation'] = merged_conversations
                except Exception as e:
                    logger.error(f"Error merging truncated plot for {book['title']}: {e}, {traceback.format_exc()}")
                    # Merge failed - save truncated plot as finished
                    u_plot['state'] = 'finished'
                    results['plots'].append(u_plot)
            else:
                # No matching plot found - mark as finished
                u_plot['state'] = 'finished'
                results['plots'].append(u_plot)

        # Separate/Update finished and truncated plots
        finished_plots = [plot for plot in plots if plot['state'] == 'finished']
        truncated_plots = [plot for plot in plots if plot['state'] == 'truncated']

        print(f"  [extract] Chunk {i} done: {len(finished_plots)} finished, {len(truncated_plots)} truncated plots")

        # Add to results
        results['chapter_beginnings'].extend(chapter_beginnings)
        results['plots'].extend(finished_plots)

    # Finish any remaining truncated plots
    for u_plot in truncated_plots:
        u_plot['state'] = 'finished'
        results['plots'].append(u_plot)
    
    results['fail_to_parse_responses'] = fail_to_parse_responses
    
    # Save results
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  [extract] Saved {len(results['plots'])} plots to {save_path}")
    return results

count_nth_generation = {i: 0 for i in range(7)}

def restore_from_cache(book):
    """
    As we typically encounter issues during the extraction process, this function restores some previously extracted plot data from cache.
    
    This function loads cached responses from extraction LLMs, processes them into the regular data format, and merges them with the results of extract(). 

    Args:
        book (dict): Book data containing title, author, and content

    Returns:
        None: Results are saved to disk, no return value
    """
    # Load existing extracted results

    save_dir = f'{args.output_dir}/extracted'
    os.makedirs(save_dir, exist_ok=True)

    with open(f'{save_dir}/{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    save_path = f'{save_dir}/{book["title"]}.json'

    # Skip if already processed
    if os.path.exists(save_path) and not args.regenerate:
       print(f"  [restore] Skipping (already exists): {save_path}")
       return 

    # Load cached API responses
    import pickle
    with open(f'.cache/cache_{book["title"]}.pkl', 'rb') as f:
        cache = pickle.load(f)
    
    # Get only the get_response cache entries
    keys = [ k for k in cache.keys() if k[0] == 'get_response' ]

    global count_nth_generation

    fail_prompts = []
    responses = {}

    # Generate chunks from book content
    chunk_generator = create_chunk_generator(book, chunk_size=8192)
    chunks = [chunk for chunk in chunk_generator]

    # Process each cached response
    for key, value in cache.items():
        if key[0] == 'get_response':
            # Extract kwargs from cache key
            dict_string = key[-1][11:-1]
            import ast
            parsed_list = ast.literal_eval(dict_string)
            restored_kwargs = dict(parsed_list)

            # Only process responses for plot extraction prompts
            if restored_kwargs['model'] == 'claude-3-5-sonnet-20240620' and (
                restored_kwargs['messages'][0]['content'].startswith("\nBased on the provided book chunk, complete the following tasks:\n\n1. Recognize chapter beginnings if")
                or restored_kwargs['messages'][0]['content'].startswith("\n请基于给定的小说正文片段，完成以下任务：\n\n1. 如果该片段中出现了新的章节开头")
            ):
                # Verify book title matches
                if not restored_kwargs['book']['title'] == book['title']:
                    logger.info(f"Warning: {restored_kwargs['book']['title']} != {book['title']}")
                    continue

                # Track generation attempts
                nth_generation = restored_kwargs['nth_generation']
                count_nth_generation[nth_generation] += 1

                # Store response
                prompt = restored_kwargs['messages'][0]['content']
                responses.setdefault(prompt, {})
                responses[prompt][nth_generation] = value

                # Track failed prompts (those that needed max retries)
                if nth_generation == 5:
                    fail_prompts.append(prompt)

    fetched_plots = []

    # Process failed prompts to extract any valid plots
    for prompt in fail_prompts:
        for nth_generation in range(6):
            if nth_generation in responses[prompt]:
                response = responses[prompt][nth_generation]
                
                # Check if response contains all required fields
                required_fields = ["chapter_beginnings", "plots", "chapter_title", "first_sentence", "last_sentence", "summary", "key_characters", "name", "description", "dialogues", "message"]
                if all(field in str(response) for field in required_fields):
                    # Extract JSON from potentially truncated response
                    from utils import extract_json
                    response = extract_json(response, post_fix_truncated_json=True)

                    if response is None:
                        continue

                    # Helper function to parse response and extract plots
                    def parse_response(response, chunk, book, **kwargs):
                        if not response:
                            return False

                        try:
                            # Normalize response format
                            if (isinstance(response, dict) and 'first_sentence' in response):
                                response = {
                                    'chapter_beginnings': [],
                                    'plots': [response],
                                    'next_chunk_start': None
                                }
                            elif isinstance(response, list):
                                plot_items = [item for item in response if isinstance(item, dict)]
                                response = {
                                    'chapter_beginnings': [],
                                    'plots': plot_items,
                                    'next_chunk_start': None
                                }
                            elif not isinstance(response, dict):
                                logger.warning(f"Unexpected response type {type(response)}, skipping")
                                return False

                            chapter_beginnings = response.get('chapter_beginnings', [])

                            plots = []

                            if response.get('next_chunk_start'):
                                response['next_chunk_start'] = find_best_match_sentence(chunk, response['next_chunk_start'])
                                remaining_chunk = build_remaining_chunk(chunk, response['next_chunk_start'])
                            else:
                                remaining_chunk = ''

                            # Process each plot in the response
                            for unprocessed_plot in response.get('plots', []):
                                chapter_title = unprocessed_plot.get('chapter_title')

                                unprocessed_plot['first_sentence'] = find_best_match_sentence(chunk, unprocessed_plot.get('first_sentence'), threshold=0.6)
                                unprocessed_plot['last_sentence'] = find_best_match_sentence(chunk, unprocessed_plot.get('last_sentence'), threshold=0.6)

                                first_sentence, last_sentence = unprocessed_plot['first_sentence'], unprocessed_plot['last_sentence']

                                original_text = extract_original_text_span(chunk, first_sentence, last_sentence)

                                # Normalize key_characters
                                if not isinstance(unprocessed_plot.get('key_characters'), list):
                                    unprocessed_plot['key_characters'] = []
                                for kc in unprocessed_plot['key_characters']:
                                    if isinstance(kc, dict) and 'name' not in kc and 'character' in kc:
                                        kc['name'] = kc.pop('character')

                                # Normalize conversations
                                normalize_plot_conversations(unprocessed_plot)

                                plot = {
                                    'text': original_text,
                                    'summary': unprocessed_plot.get('summary', ''),
                                    'prominence': unprocessed_plot.get('prominence', 50),
                                    'key_characters': unprocessed_plot['key_characters'],
                                    'chapter': chapter_title,
                                    'conversation': unprocessed_plot['conversation'],
                                    'state': unprocessed_plot.get('state', 'finished')
                                }

                                plots.append(plot)

                            print_json(response)
                            logger.info(json.dumps(response, ensure_ascii=False, indent=2))

                            return chapter_beginnings, plots, remaining_chunk

                        except Exception as e:
                            logger.error(f"Error processing chunk for book {book['title']}: {e}, {traceback.format_exc()}")
                            return False
                    
                    # Extract chunk from prompt
                    chunk = prompt.split('==Truncated plot from previous chunk (to be finished)==')[0].split('==Chunk of Book Content==')[-1].strip(' \n')

                    # Parse response to get plots
                    res = parse_response(response, chunk, book)

                    if res :
                        chapter_beginnings, plots, remaining_chunk = res
                    else:
                        continue

                    # Process extracted plots
                    for plot in plots:
                        plot['state'] = 'finished'
                        plot['i_chunk'] = -1
                        # Find which chunk this plot belongs to
                        for i_chunk, another_chunk in enumerate(chunks):
                            if another_chunk.strip(' \n').endswith(chunk[-100:]):
                                plot['i_chunk'] = i_chunk
                                break

                    fetched_plots.extend(plots)
                    break 
    
    # Find chunk indices for original plots
    for plot in results['plots']:
        plot['i_chunk'] = -1
        for i_chunk, chunk in enumerate(chunks):
            if plot['text'][-100:] in chunk:
                plot['i_chunk'] = i_chunk
                break

    # Merge and sort all plots
    logger.info(f'Number of Original Plots: {len(results["plots"])}, Fetched New Plots: {len(fetched_plots)}, Total Plots: {len(results["plots"]) + len(fetched_plots)}')

    new_plots = results['plots'] + fetched_plots
    new_plots = sorted(new_plots, key=lambda x: x['i_chunk'])

    results['plots'] = new_plots

    # Save restored results (together with the original results)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return 

def assemble(book):
    """
    Assemble the extracted plot data for a book into the final structured data.
    
    This function:
    1. Processes extracted plots and conversations
    2. Enhances conversation scenarios and character motivations using LLM
    3. Normalizes and standardizes character names
    4. Generates character profiles and datasets
    5. Saves the final assembled data
    
    Args:
        book (dict): Book data containing title, author and content
        
    Returns:
        None: Results are saved to disk at {args.output_dir}/final/{book_title}.json
    """
    # Set up caching for this book
    from utils import set_cache_path
    set_cache_path(f'.cache/cache_{book["title"]}.pkl')
    
    os.makedirs(f'{args.output_dir}/final', exist_ok=True)

    save_path = f'{args.output_dir}/final/{book["title"]}.json'

    # Skip if already processed
    if os.path.exists(save_path) and not args.regenerate:
        print(f"  [assemble] Skipping (already exists): {save_path}")
        return
    
    # Load extracted plot data
    with open(f'{args.output_dir}/extracted/{book["title"]}.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    plots = results['plots']
    print(f"  [assemble] Loaded {len(plots)} plots, language detection...")

    # Detect language from first plot text
    if len(plots) > 0:
        from utils import lang_detect
        language = lang_detect(plots[0]['text'][:100])
        language = {'zh': '中文', 'en': '英文'}.get(language, '英文')
    else:
        language = '英文'

    # Normalize conversation format and enhance scenarios/motivations
    print(f"  [assemble] Enhancing {len(plots)} plots' scenarios/motivations...")
    for i_plot, plot in enumerate(plots):
        # Normalize all conversations upfront
        normalize_plot_conversations(plot)

        if not plot['conversation']:
            continue 
        for conversation in plot['conversation']:
            # Prepare input for conversation enhancement
            input_conversation = {
                'plot_summary': plot['summary'], 
                'character_information': plot['key_characters'],
                **conversation
            }
            
            # Get character names for this conversation
            conv_key_characters = [
                _.get('name', _.get('character', '')) 
                for _ in conversation['key_characters'] 
                if 'name' in _ or 'character' in _
            ]

            # Generate prompt for enhancing scenario and motivations
            prompt = f"""
下面给你一段来自《{book['title']}》的对话及其上下文信息。请你补强 scene setup 和人物 motivation，使其更适合作为中文角色扮演或戏剧表演的背景设定：

1. 认真阅读给定的对话、剧情摘要与人物信息。
2. 扩写 `scenario`，补充演员理解场景所需的关键信息，包括背景、氛围、人物关系、局势张力等，但不要提前泄露后续对话中才会揭示的信息。
3. 补强每个角色的 `motivation`，写出其在对话开始前完整的心理和情绪状态，包括感受、想法、目的、想谈的话题、想传达的信息等，并且要符合人物既有设定。

===输出格式===
请严格按照以下 JSON 格式输出：
{{
    "scenario": "详细的中文场景说明（建议 200 字以内），为演员提供必要的背景和氛围信息，但不要泄露后续对话才会呈现的内容。",
    "key_characters": [
        {{
            "name": "角色姓名",
            "motivation": "该角色在对话开始前完整的心理与情绪状态（建议 100 字以内），包括其感受、动机、目的，以及想表达或讨论的信息。必须使用中文。"
        }}
    ],
}}

===要求===
1. 必须严格遵守上述 JSON 格式。
2. [重要] 所有字符串中的双引号都必须正确转义。
3. 输出时角色姓名必须与输入保持完全一致，不要改名，也不要把“环境”改成英文。
4. 输出语言必须与输入一致。当前输入是中文时，`scenario` 和 `motivation` 必须全部使用中文。
5. 除非输入中本来就有英文，否则不要输出英文句子、英文 thought 或英文舞台说明。
6. `key_characters` 必须与输入中的角色完全一致，包括 {conv_key_characters}。

===输入对话与背景===
{json.dumps(input_conversation, ensure_ascii=False, indent=2)}
"""

            # Helper function to validate enhanced conversation response
            from utils import extract_json
            def parse_response(response, characters, **kwargs):
                try:
                    assert 'scenario' in response 
                    assert 'key_characters' in response
                    key_characters = {_['name']: _['motivation'] for _ in response['key_characters']}
                    for character in characters:
                        assert character in key_characters
                    return response
                except:
                    return False

            # Get enhanced conversation from LLM
            print(f"  [assemble] Enhancing plot {i_plot+1}/{len(plots)}, conversation {conversation.get('topic', '?')[:30]}...")
            response = get_response_json(
                [extract_json, parse_response], 
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                characters=conv_key_characters,
                max_retry=5
            )
            
            # Normalize character name field
            try:
                for chara in response['key_characters']:
                    if 'name' not in chara and 'character' in chara:
                        chara['name'] = chara.pop('character')
            except:
                continue
                
            # Update conversation with enhanced content
            conversation['scenario'] = response['scenario']
            enhanced_motivations = {chara['name']: chara['motivation'] for chara in response['key_characters']}
            for chara in conversation['key_characters']:
                if 'name' not in chara and 'character' in chara:
                    chara['name'] = chara.pop('character')
                chara['motivation'] = enhanced_motivations[chara['name']]

    # Filter out plots with invalid character data
    plots = [
        p for p in plots 
        if all(['name' in c for c in p['key_characters']]) 
        and len(p['key_characters']) == len(set([c['name'] for c in p['key_characters']]))
    ]

    # Split plots into train/test sets
    split_index = int(len(plots) * 0.9)

    # Collect all character names
    print(f"  [assemble] Standardizing character names...")
    character_names = set()
    for plot in plots:
        # Add names from plot key characters
        for character in plot['key_characters']:
            character_names.add(character['name'])

        # Normalize conversation format (already done earlier, but ensure again after filtering)
        normalize_plot_conversations(plot)

        # Process each conversation
        for conversation in plot['conversation']:
            # Add names from conversation key characters
            for key_character in conversation.get('key_characters', []):
                name = key_character.get('name')
                if name:
                    character_names.add(name)

            # Add names from dialogue utterances
            for utterance in conversation.get('dialogues', []):
                char_name = utterance.get('character')
                if char_name:
                    character_names.add(char_name)

    # Sort character names
    character_names = sorted(character_names)

    # Generate prompt for standardizing character names
    prompt = """给定一组人物名字、称谓或指代方式，请完成以下任务：
1. 识别其中真正具名的人物，并给出他们的正式姓名列表（使用 {language}）。
2. 对输入列表中的每个名字或称呼，判断它是否指向某个具名角色：
   - 如果是，请映射到该角色的正式姓名；
   - 如果不是具名角色，而只是泛称、旁白标签、环境标签或无法确认身份的称呼，请标记为 "impersonal"。

===输出格式===
请严格按照以下 JSON 格式输出：
{{
    "named_characters": [
        具名角色的正式姓名列表，每个角色只出现一次。
    ],
    "to_official_name": {{
        "输入列表中的名字或称呼": "对应角色的正式姓名；如果不是具名角色，则输出 'impersonal'"
    }}
}}
===输入===
{character_names}
"""

    prompt = prompt.replace('{character_names}', str(character_names))
    prompt = prompt.replace('{language}', language)

    # Helper function to validate name standardization response
    def parse_response(response, **kwargs):
        if 'named_characters' in response:
            return response
        else:
            return False

    # Get standardized names from LLM
    from utils import extract_json
    response = get_response_json(
        [extract_json, parse_response], 
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_retry=5
    )

    # Extract official names, falling back to candidate model if needed
    try:
        official_names = response['named_characters']
    except:
        response = get_response_json(
            [extract_json, parse_response],
            model=args.candidate_model,
            messages=[{"role": "user", "content": prompt}],
            max_retry=5
        )
        official_names = response['named_characters']
    
    # Normalize list values in name mapping
    for k, v in response['to_official_name'].items():
        if isinstance(v, list):
            response['to_official_name'][k] = v[0]

    # Add missing official names
    official_names += [
        n for n in set(response['to_official_name'].values()) - set(official_names)
        if n.lower() not in ['impersonal', 'environment'] and n not in ['环境']
    ]

    to_official_name = response['to_official_name']

    # Handle names missing from the mapping
    missing_names = []
    for name in character_names:
        if name not in to_official_name:
            print(f"Warning: {name} not included in to_official_name")
            missing_names.append(name)

    print(f"Missing names: {missing_names}")

    # Find closest matches for missing names
    for name in missing_names:
        closest_name = find_best_match_passage(official_names, name)
        print(f"Closest name: {closest_name}")
        if closest_name != -1:
            to_official_name[name] = official_names[closest_name]
        else:
            to_official_name[name] = "impersonal"

    # Initialize character datasets
    character_datasets = {
        character: {
            "plots": [],
            "conversations": [],
            "utterances": []
        } for character in official_names
    }

    # Populate character datasets
    for i_p, plot in enumerate(plots):
        # Process plot key characters
        for character in plot['key_characters']:
            char_name = character.get('name')
            if not char_name or char_name not in to_official_name:
                continue
            if to_official_name[char_name] != "impersonal":
                character['name'] = to_official_name[char_name]
                character_datasets[character['name']]['plots'].append((i_p, character))

        # Process conversations
        for i_c, conversation in enumerate(plot.get('conversation', [])):
            # Process conversation key characters
            for character in conversation.get('key_characters', []):
                char_name = character.get('name')
                if not char_name or char_name not in to_official_name:
                    continue
                if to_official_name[char_name] != "impersonal":
                    character['name'] = to_official_name[char_name]
                    character_datasets[character['name']]['conversations'].append((i_p, i_c, character))

            # Process utterances
            for i_u, utterance in enumerate(conversation.get('dialogues', [])):
                char_key = utterance.get('character')
                if not char_key or char_key not in to_official_name:
                    continue
                if to_official_name[char_key] != "impersonal":
                    utterance['character'] = to_official_name[char_key]
                    character_datasets[utterance['character']]['utterances'].append((i_p, i_c, i_u, utterance))

    # Generate character profiles prompt template
    prompt = """请为《{book_title}》中的角色 {character_name} 生成一段简洁、连贯、叙述式的人物小传。

这段人物小传应像角色指南中的正式介绍，尽量自然地整合以下信息：角色背景、外貌或气质特征、性格特点、核心动机、重要关系、关键经历、主要剧情参与情况、关键决定与行为、人物成长或变化，以及其他有助于理解该角色的重要信息。

请使用 {language} 写作，风格要求简洁但信息密度高，帮助读者快速把握该角色在作品中的意义。不要编造不确定的信息；若资料不足，就只写能够从已知信息中确定的内容。

下面提供的是该角色相关的部分剧情摘要和对话，供你参考：

{character_data}

现在请生成角色小传，并以 `===Profile===` 作为开头。
"""

    # Generate character profiles
    print(f"  [assemble] Generating profiles for {len(character_datasets)} characters...")
    for character_name, character_data in character_datasets.items():
        print(f"  [assemble] Generating profile for: {character_name}")
        # Get plots involving this character
        involved_plots = sorted(set(
            [p[0] for p in character_data['plots']] + 
            [c[0] for c in character_data['conversations']] + 
            [u[0] for u in character_data['utterances']]
        ))

        # Filter to training set plots only
        involved_plots = [i_p for i_p in involved_plots if i_p < split_index]
        
        # Collect plot information
        plot_infos = []
        for i_p in involved_plots:
            plot = plots[i_p]

            plot_info = {
                "plot": plot['summary'],
            }

            # Add character-specific information
            for key_character_info in plot['key_characters']:
                if key_character_info['name'] == character_name:
                    plot_info["character_experience"] = key_character_info

            # Add relevant conversations
            plot_info["conversation"] = []
            for conversation in plot.get('conversation', []):
                conv_char_names = [kc.get('name', '') for kc in conversation.get('key_characters', [])]
                dialogue_char_names = [u.get('character', '') for u in conversation.get('dialogues', [])]
                if character_name in conv_char_names or character_name in dialogue_char_names:
                    plot_info["conversation"].append(conversation)
            
            plot_infos.append(plot_info)

        

        character_prompt = prompt.replace("{character_name}", character_name).replace("{book_title}", book["title"]).replace("{character_data}", json.dumps(plot_infos, ensure_ascii=False, indent=2)).replace("{language}", language)

        print(character_prompt)

        # Get profile from LLM with retries
        nth_generation = 0
        while True:
            if nth_generation > 0:
                profile = get_response(
                    model=args.model,
                    messages=[{"role": "user", "content": character_prompt}],
                    nth_generation=nth_generation
                )
            else:
                profile = get_response(
                    model=args.model,
                    messages=[{"role": "user", "content": character_prompt}]
                )

            try:
                profile = profile.split("===Profile===", 1)[1].strip() 
                if profile.startswith('I apologize'): profile = ''
                character_datasets[character_name]['profile'] = profile
                break
            except:
                nth_generation += 1
                if nth_generation > 5:
                    character_datasets[character_name]['profile'] = ''
                    break
                continue

    # Update data format for readability
    for character_name, character_data in character_datasets.items():
        # Flatten plot data
        for i, plot in enumerate(character_data['plots']):
            character_data['plots'][i] = plot[-1]
            character_data['plots'][i]['i_p'] = plot[0]
        
        # Flatten conversation data
        for i, conversation in enumerate(character_data['conversations']):
            character_data['conversations'][i] = conversation[-1]
            character_data['conversations'][i]['i_p'] = conversation[0]
            character_data['conversations'][i]['i_c'] = conversation[1]
        
        # Flatten utterance data
        for i, utterance in enumerate(character_data['utterances']):
            character_data['utterances'][i] = utterance[-1]
            character_data['utterances'][i]['i_p'] = utterance[0]
            character_data['utterances'][i]['i_c'] = utterance[1]
            character_data['utterances'][i]['i_u'] = utterance[2]

    # Save final results
    results['character_datasets'] = character_datasets
    results['split_plot_index'] = split_index

    results.pop("chapter_beginnings")
    results.pop("fail_to_parse_responses")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"  [assemble] Saved final data to {save_path}")
if __name__ == '__main__':

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read input data
    books_data = read_jsonl_records(args.input)
    validate_input_books(books_data, args.input)

    # Clean book titles
    for book in books_data:
        book['title'] = book['title'].replace('/', '-').replace(':', '_').replace('.', ' ')

    print(f"\n{'='*60}")
    print(f"[START] CoSER Data Construction Pipeline")
    print(f"  Input: {args.input}")
    print(f"  Output root: {args.base_output_dir}")
    print(f"  Output run: {args.output_dir}")
    print(f"  Run ID: {args.run_timestamp}")
    print(f"  Model: {args.model}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Input contract: prepared JSONL records (no preprocess stage)")
    print(f"  Books: {len(books_data)}")
    print(f"  Book titles: {[b['title'] for b in books_data]}")
    print(f"{'='*60}\n")

    logger.info(f"Processing {len(books_data)} books")

    def process_book(book):
        title = book.get('title', 'Unknown')
        try:
            print(f"\n{'='*60}")
            print(f"[BOOK] Starting: {title}")
            print(f"{'='*60}")

            print(f"\n[STAGE 1/3] Extracting plots from: {title}")
            extract(book)
            print(f"[STAGE 1/3] Extraction complete: {title}")

            print(f"\n[STAGE 2/3] Restoring from cache: {title}")
            restore_from_cache(book)
            print(f"[STAGE 2/3] Cache restore complete: {title}")

            print(f"\n[STAGE 3/3] Assembling final data: {title}")
            result = assemble(book)
            print(f"[STAGE 3/3] Assembly complete: {title}")

            print(f"\n[BOOK] Successfully processed: {title}")
            logger.info(f"Successfully processed book: {title}")
            return result
        except Exception as e:
            print(f"\n[BOOK] FAILED: {title} - {str(e)}")
            logger.error(f"Error processing book {title}: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    if args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        
        logger.info(f"Starting parallel processing with {args.num_workers} workers")

        # Process books in parallel
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            processed_books = list(tqdm(
                executor.map(process_book, books_data),
                total=len(books_data),
                desc="Processing books"
            ))
    else:
        processed_books = []
        for book in tqdm(books_data):
            processed_book = process_book(book)
            processed_books.append(processed_book)











