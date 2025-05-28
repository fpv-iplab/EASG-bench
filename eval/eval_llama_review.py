import argparse
import json
import os
import numpy as np
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LLM:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    @torch.inference_mode()
    def prompt_text(self, text_prompt, max_new_tokens):
        setup_seed(42)
        messages = [
            {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
            {"role": "user", "content": text_prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1]:]
        msg = self.tokenizer.decode(response, skip_special_tokens=True)
        return msg

model = LLM('meta-llama/Meta-Llama-3-8B-Instruct')


def get_eval(content: str, max_tokens: int):
    response = model.prompt_text(content, max_tokens)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Llama-based QA evaluation.')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer')
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('-s', '--scale', default="")
    parser.add_argument('--max-tokens', type=int, default=2048, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    # rule
    with open(args.rule, 'r') as f:
        prompt = f.read()

    # answer and question
    answers = json.load(open(args.answer, 'r'))

    # context
    contexts = json.load(open(args.context, 'r'))

    # scaling factor
    scale = {}
    if args.scale:
        scale = json.load(open(args.scale, 'r'))

    out_list = []
    for answer in tqdm(answers):

        ques = answer['question']
        ans1 = answer['answer']
        ans2 = answer['prediction']

        cap_str = contexts.get(answer['vid'])
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques}\n\n'
                   f'[Groundtruth]\n{ans1}\n\n[End of Groundtruth]\n\n'
                   f'[Assistant]\n{ans2}\n\n[End of Assistant]\n\n'
                   f'[System]\n{prompt}\n\n')
        
        review = get_eval(content, args.max_tokens)
        review = review.replace('Score:', '').strip()
        try:
            scores, review = review.split('\n\n')
            scores = int(scores)
        except:
            print("error", review)
            scores = 0

        answer.update({"review": review, "scores": scores})
        print("==============================")
        print(f"Q: {ques}\n")
        print(f"GT: {ans1}\n")
        print(f"A: {ans2}\n")
        print(f"Score:\n{scores}\n\n{review}")
        out_list.append(answer)

    with open(args.answer, 'w') as f:
        json.dump(out_list, f)

    scores = {}
    qtypes = ['purpose', 'direct', 'indirect', 'before', 'after']
    for qtype in qtypes:
        scores[qtype] = np.mean([x['scores'] for x in out_list if x['type'] == qtype]) / scale.get(qtype, 1.0)
        print(f"{qtype}: {scores[qtype] * 100:.2f}")
    
    with open(args.output, 'w') as f:
        json.dump(scores, f)


