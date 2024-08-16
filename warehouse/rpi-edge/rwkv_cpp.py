import argparse
import copy
import json
import os
import pathlib
import time
import sampling

from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from tokenizer_util import add_tokenizer_argument, get_tokenizer
from typing import List, Dict, Optional

DOUBLE_END_OF_LINE_TOKEN: int = 535
END_OF_LINE_TOKEN: int = 187
END_OF_TEXT_TOKEN: int = 0
FREQUENCY_PENALTY: float = 0.2
MAX_CONTEXT_LENGTH: int = 4096
MAX_GENERATION_LENGTH: int = 250
PRESENCE_PENALTY: float = 0.2
TEMPERATURE: float = 0.7
TOP_P: float = 0.3

parser = argparse.ArgumentParser(description='Provide terminal-based chat interface for RWKV model')
parser.add_argument('model_path', help='Path to RWKV model in ggml format')

add_tokenizer_argument(parser)
args = parser.parse_args()

script_dir: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent

prompt_data = json.loads("""
{
	"user": "User",
	"bot": "Assistant",
	"separator": ":",
	"prompt": "You are a helpful assistant. Provide accurate, one-sentence answers."
}
""")

user, bot, separator, init_prompt = prompt_data['user'], prompt_data['bot'], prompt_data['separator'], prompt_data[
    'prompt']

if init_prompt == '':
    raise ValueError('Prompt must not be empty')

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')
model = rwkv_cpp_model.RWKVModel(library, args.model_path)

tokenizer_decode, tokenizer_encode = get_tokenizer(args.tokenizer, model.n_vocab)

processed_tokens: List[int] = []
logits: Optional[rwkv_cpp_model.NumpyArrayOrPyTorchTensor] = None
state: Optional[rwkv_cpp_model.NumpyArrayOrPyTorchTensor] = None


def process_tokens(_tokens: List[int], new_line_logit_bias: float = 0.0) -> None:
    global processed_tokens, logits, state

    logits, state = model.eval_sequence_in_chunks(_tokens, state, state, logits, use_numpy=True)

    processed_tokens += _tokens

    logits[END_OF_LINE_TOKEN] += new_line_logit_bias


state_by_thread: Dict[str, Dict] = {}


def save_thread_state(_thread: str) -> None:
    state_by_thread[_thread] = {
        'tokens': copy.deepcopy(processed_tokens),
        'logits': copy.deepcopy(logits),
        'state': copy.deepcopy(state)
    }


def load_thread_state(_thread: str) -> None:
    global processed_tokens, logits, state

    thread_state = state_by_thread[_thread]

    processed_tokens = copy.deepcopy(thread_state['tokens'])
    logits = copy.deepcopy(thread_state['logits'])
    state = copy.deepcopy(thread_state['state'])


def split_last_end_of_line(tokens: List[int]) -> List[int]:
    if len(tokens) > 0 and tokens[-1] == DOUBLE_END_OF_LINE_TOKEN:
        tokens = tokens[:-1] + [END_OF_LINE_TOKEN, END_OF_LINE_TOKEN]
    return tokens


processing_start: float = time.time()

prompt_tokens = tokenizer_encode(init_prompt)
prompt_token_count = len(prompt_tokens)
print(f'Processing {prompt_token_count} prompt tokens, may take a while')

process_tokens(split_last_end_of_line(prompt_tokens))

processing_duration: float = time.time() - processing_start

print(f'Processed in {int(processing_duration)} s, {int(processing_duration / prompt_token_count * 1000)} ms per token')

save_thread_state('chat_init')
save_thread_state('chat')

print(f'\nYour name is {user}. Write something and press Enter.')

while True:
    user_input: str = input()

    if user_input.lower() == "exit":
        print("Exiting chat.")
        break

    msg: str = user_input.replace('\\n', '\n').strip()
    temperature: float = TEMPERATURE
    top_p: float = TOP_P

    if '-temp=' in msg:
        temperature = float(msg.split('-temp=')[1].split(' ')[0])
        msg = msg.replace('-temp=' + f'{temperature:g}', '')

        if temperature <= 0.2:
            temperature = 0.2

        if temperature >= 5:
            temperature = 5

    if '-top_p=' in msg:
        top_p = float(msg.split('-top_p=')[1].split(' ')[0])
        msg = msg.replace('-top_p=' + f'{top_p:g}', '')

        if top_p <= 0:
            top_p = 0

    msg = msg.strip()

    if msg == '+reset':
        load_thread_state('chat_init')
        save_thread_state('chat')
        print(f'\n{bot}{separator} Chat reset.\n')
        continue

    load_thread_state('chat')
    new = f'{user}{separator} {msg}\n\n{bot}{separator}'
    process_tokens(tokenizer_encode(new), new_line_logit_bias=-999999999)

    if len(processed_tokens) > MAX_CONTEXT_LENGTH:
        processed_tokens = processed_tokens[-MAX_CONTEXT_LENGTH:]

    save_thread_state('chat_pre')

    thread = 'chat'

    start_index: int = len(processed_tokens)
    accumulated_tokens: List[int] = []
    token_counts: Dict[int, int] = {}

    for i in range(MAX_GENERATION_LENGTH):
        for n in token_counts:
            logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

        token: int = sampling.sample_logits(logits, temperature, top_p)

        if token == END_OF_TEXT_TOKEN:
            print()
            break

        if token not in token_counts:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

        process_tokens([token])
        accumulated_tokens += [token]

        decoded: str = tokenizer_decode(accumulated_tokens)

        if f'{user}{separator}' in decoded:
            print("\n[ERROR] Detected user input in bot response. Regenerating...")
            break

        if thread == 'chat':
            if '\n\n' in tokenizer_decode(processed_tokens[start_index:]):
                break

        if i == MAX_GENERATION_LENGTH - 1:
            print()

    full_response = tokenizer_decode(accumulated_tokens).strip()

    if full_response:
        print(f"{bot}{separator} {full_response}")

        print(flush=True)

        os.system(f'espeak-ng -v en-us -s 200 "{full_response}"')

    save_thread_state(thread)
