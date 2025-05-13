import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Dict, Sequence

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
)

from transformers.hf_argparser import HfArg
import json

@dataclass
class Arguments(TrainingArguments):
    model_name_or_path: str = HfArg(
        default = '', # Replace with the path to your model
        help="The model name or path, e.g., `meta-llama/Llama-2-7b-hf`",
    )

    dataset: str = HfArg(
        default = 'dataset/alpaca_data.json',
        help="Setting the names of data file.",
    )

    model_max_length: int = HfArg(
        default=2048,
        help="The maximum sequence length",
    )

    save_only_model: bool = HfArg(
        default=True,
        help="When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state.",
    )

    bf16: bool = HfArg(
        default = False,
        help="Whether to use bf16 (mixed) precision instead of 32-bit.",
    )

    output_dir: str = HfArg(
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

class SFTDataset:
    IGNORE_INDEX = -100
    
    instruction_template = "\n### Instruction:\n"
    response_template = "\n### Output:\n"
    format_template = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. " +
            "Write a response that appropriately completes the request." + instruction_template + "{instruction}" + "\n" +
            "{input}" + response_template
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. " +
            "Write a response that appropriately completes the request." + instruction_template + "{instruction}" +
            response_template
        ),
    }

    def __init__(self, args, tokenizer):
        self.args = args
        self.block_size = self.args.model_max_length
        self.tokenizer = tokenizer
        self.input_ids, self.labels = self.process(self.tokenizer)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def encode_src_tgt(self, s, t, tokenizer):
        source_id = tokenizer.encode(s, truncation=True, max_length=self.model_max_length)
        
        tokenizer.add_eos_token = True
        input_id = tokenizer.encode(s + t, truncation=True, max_length=self.model_max_length)
        
        tokenizer.add_eos_token = False
        label = input_id.clone()
        label[:len(source_id)] = self.IGNORE_INDEX
        
        return input_id, label

    def process(self, tokenizer):
        input_ids = []
        labels = []
        list_data_dict = json.load(open(self.args.dataset))

        for example in list_data_dict:
            if example.get('input', ''):
                s = self.format_template['prompt_input'].format(
                    instruction=example['instruction'],
                    input=example['input']
                )
            else:
                s = self.format_template['prompt_no_input'].format(
                    instruction=example['instruction']
                )

            example['response'] = example.pop('output')
            t = example['response'].strip()

            input_id, label = self.encode_src_tgt(s, t, tokenizer)

            input_ids.append(input_id)
            labels.append(label)

        return input_ids, labels

@dataclass
class DataCollatorForSupervisedDataset():
    tokenizer: PreTrainedTokenizer
    IGNORE_INDEX = -100

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=self.IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
        )

def main():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # 设置一些必要的参数
    args.framework = "pt"
    args.use_cpu = True
    args.no_cuda = True
    args.use_mps_device = False
    args.per_device_train_batch_size = 4
    args.num_train_epochs = 3
    args.learning_rate = 2e-5
    args.save_strategy = "epoch"
    args.logging_steps = 10

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        add_eos_token=False,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    kwargs = dict(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=SFTDataset(args, tokenizer),
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
    )

    trainer = Trainer(**kwargs)
    trainer.train()
    trainer.save_model(args.output_dir + "/checkpoint-final")
    trainer.save_state()

if __name__ == "__main__":
    main() 