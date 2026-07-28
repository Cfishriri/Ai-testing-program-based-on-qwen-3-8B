import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChatBot:
    def __init__(self, args):
        self.max_new_tokens = args.max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="auto",
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        self.messages = []
        if args.system_prompt:
            self.messages.append({"role": "system", "content": args.system_prompt})

    def reply(self, message):
        self.messages.append({"role": "user", "content": message})
        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        response_tokens = generated[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        self.messages.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        system_messages = [item for item in self.messages if item["role"] == "system"]
        self.messages = system_messages


def parse_args():
    parser = argparse.ArgumentParser(description="Start an interactive persistent Qwen chat.")
    parser.add_argument("--model-path", required=True, help="Local model directory or Hugging Face model ID.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    bot = ChatBot(parse_args())
    print("Chat ready. Commands: /reset, /exit")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in {"/exit", "exit", "quit"}:
            break
        if message.lower() == "/reset":
            bot.reset()
            print("Conversation reset.")
            continue
        print(f"Assistant: {bot.reply(message)}")


if __name__ == "__main__":
    main()
