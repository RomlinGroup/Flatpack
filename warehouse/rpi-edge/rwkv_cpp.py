# Code adapted from https://github.com/RWKV/rwkv.cpp/blob/master/python/chat_with_bot.py (MIT license)

import argparse
import copy
import os
import sampling
import time

from rwkv_cpp import rwkv_cpp_model, rwkv_cpp_shared_library
from tokenizer_util import add_tokenizer_argument, get_tokenizer
from typing import Dict, List, Optional

END_OF_TEXT_TOKEN: int = 0
FREQUENCY_PENALTY: float = 0.2
MAX_GENERATION_LENGTH: int = 256
PRESENCE_PENALTY: float = 0.2
TEMPERATURE: float = 0.7
TOP_P: float = 0.5

parser = argparse.ArgumentParser(
    description='Generate some text with an RWKV model'
)

parser.add_argument(
    'model_path',
    help='Path to RWKV model in ggml format'
)

parser.add_argument(
    '--prompt',
    default="You are an AI assistant. Answer the questions helpfully.",
    help='Text prompt to start the generation',
    type=str
)

add_tokenizer_argument(parser)
args = parser.parse_args()

library = rwkv_cpp_shared_library.load_rwkv_shared_library()
model = rwkv_cpp_model.RWKVModel(library, args.model_path)
tokenizer_decode, tokenizer_encode = get_tokenizer(args.tokenizer, model.n_vocab)

processed_tokens: List[int] = []

prompt: str = args.prompt
prompt_tokens: List[int] = tokenizer_encode(prompt)

logits: Optional[rwkv_cpp_model.NumpyArrayOrPyTorchTensor] = None
state: Optional[rwkv_cpp_model.NumpyArrayOrPyTorchTensor] = None


def process_tokens(_tokens: List[int], new_line_logit_bias: float = 0.0) -> None:
    global processed_tokens, logits, state

    logits, state = model.eval_sequence_in_chunks(_tokens, state, state, logits, use_numpy=True)
    processed_tokens += _tokens

    logits[187] += new_line_logit_bias


def chat() -> None:
    global processed_tokens, logits, state
    print('Starting chat... Type your message and press Enter. Type "exit" to quit.')

    while True:
        user_input: str = input("> ").strip()

        if user_input.lower() == "exit":
            print("Exiting chat.")
            break

        msg = f"User: {user_input}\nBot:"
        msg_tokens = tokenizer_encode(msg)
        process_tokens(msg_tokens)

        print("Bot: ", end='')

        accumulated_tokens: List[int] = []
        bot_response: str = ""
        token_counts: Dict[int, int] = {}

        for i in range(MAX_GENERATION_LENGTH):
            for n in token_counts:
                logits[n] -= PRESENCE_PENALTY + token_counts[n] * FREQUENCY_PENALTY

            token: int = sampling.sample_logits(logits, TEMPERATURE, TOP_P)

            if token == END_OF_TEXT_TOKEN:
                break

            token_counts[token] = token_counts.get(token, 0) + 1

            process_tokens([token])
            accumulated_tokens.append(token)

            decoded: str = tokenizer_decode(accumulated_tokens)

            if '\uFFFD' not in decoded:
                bot_response += decoded
                accumulated_tokens = []

        if bot_response.strip() == "":
            bot_response = "I'm sorry, I didn't catch that. Could you please rephrase?"

        print(bot_response.strip(), flush=True)

        os.system(f'espeak-ng -v en-us -s 150 -p 70 -g 10 "{bot_response.strip()}"')


if __name__ == "__main__":
    process_tokens(prompt_tokens)
    chat()
    model.free()
