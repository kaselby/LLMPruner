
import torch
import torch.nn as nn
import numpy as np
import tqdm
import logging
import copy
import gc
import os

from pathlib import Path

import src.torch_pruning as tp 
from src.pruner import hf_llama_pruner as llama_pruner
from src.utils.logger import LoggerWithDepth
from src.datasets.example_samples import get_examples
from src.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

'''
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()
'''

def _prune_blockwise(
    model, 
    tokenizer, 
    imp, 
    forward_prompts,
    logger, 
    pruner_type='l2',
    taylor='param_first',
    global_pruning=False,
    iterative_steps=1,
    pruning_ratio=0.5,
    block_attention_layer_start=3,
    block_attention_layer_end=31,
    block_mlp_layer_start=3,
    block_mlp_layer_end=31,
    num_examples=10,
    device=torch.device('cuda')
):
    kwargs = {
        "importance": imp,
        "global_pruning": global_pruning,
        "iterative_steps": iterative_steps,
        "ch_sparsity": pruning_ratio, 
        "ignored_layers":[],
        "channel_groups": {
        },
        "consecutive_groups": {
            layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers
        },
        "customized_pruners": {
            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
        },
        "root_module_types": None, 
        "root_instances": [model.model.layers[i].self_attn.q_proj for i in range(block_attention_layer_start, block_attention_layer_end)] +
                            [model.model.layers[i].mlp.gate_proj for i in range(block_mlp_layer_start, block_mlp_layer_end)]
    }
    logger.info("Pruning Attention Layer = {}".format(list(range(block_attention_layer_start, block_attention_layer_end))))
    logger.info("Pruning MLP Layer = {}".format(list(range(block_mlp_layer_start, block_mlp_layer_end))))

    pruner = tp.pruner.MetaPruner(
        model,
        forward_prompts,
        **kwargs
    )
    model.zero_grad()

    logger.info("Start Pruning")
    for i in range(iterative_steps):
        if pruner_type in ['taylor']:
            example_prompts = get_examples('bookcorpus', tokenizer, num_examples, seq_len = 64).to(device)
            logger.info("Start Backwarding in iterative steps = {}...".format(i))
            if taylor in ['param_mix', 'param_second']:
                for j in range(num_examples):
                    batch_input = example_prompts[j].unsqueeze(0)
                    loss = model(batch_input, labels=batch_input).loss
                    logger.info("Loss = {}".format(loss))
                    loss.backward()

                    for module_param in model.parameters():
                        module_param.grad = module_param.grad * module_param.grad / num_examples
                        if hasattr(module_param, 'acc_grad'):
                            module_param.acc_grad += module_param.grad
                        else:
                            module_param.acc_grad = copy.deepcopy(module_param.grad)
                    model.zero_grad()
                    del loss.grad
                
            loss = model(example_prompts, labels=example_prompts).loss
            logger.info("Loss = {}".format(loss))
            loss.backward()

        pruner.step()

        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("After Iter {}/{}, #parameters: {}".format(i+1, iterative_steps, after_pruning_parameters))
    
        # modify inference-related attributes
        for layer in model.model.layers:
            layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

    # Clean the gradient in the model
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None


def _prune_channelwise(        
    model, 
    tokenizer, 
    imp, 
    forward_prompts,
    logger, 
    pruner_type='l2',
    taylor='param_first',
    global_pruning=False,
    iterative_steps=1,
    pruning_ratio=0.5,
    block_attention_layer_start=3,
    block_attention_layer_end=31,
    block_mlp_layer_start=3,
    block_mlp_layer_end=31,
    num_examples=10,
    device=torch.device('cuda')
):
    kwargs = {
        "importance": imp,
        "global_pruning": global_pruning,
        "iterative_steps": iterative_steps,
        "ch_sparsity": pruning_ratio, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        "ignored_layers":[],
        #"round_to": model.config.num_attention_heads * 2,
        "channel_groups": {
            #layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
        },
        "customized_pruners": {
            LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
            #LlamaAttention: llama_pruner.hf_attention_pruner,
        },
        "root_module_types": [LlamaRMSNorm, LlamaAttention],
    }

    pruner = tp.pruner.MetaPruner(
        model,
        forward_prompts,
        **kwargs
    )
    model.zero_grad()
    
    logger.info("Start Pruning")
    for i in range(iterative_steps):

        if pruner_type in ['taylor']:
            example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len = 64)
            logger.info("Start Backwarding in iterative steps = {}...".format(i))
            loss = model(example_prompts, labels=example_prompts).loss
            logger.info("Loss = {}".format(loss))
            loss.backward()

        pruner.step()

        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("After Iter {}/{}, #parameters: {}".format(i+1, iterative_steps, after_pruning_parameters))

    # Clean the gradient in the model
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None

    # modify inferece-related attributes
    model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
    model.zero_grad()


def prune_llama2(
    model, 
    tokenizer, 
    grouping_strategy='sum',
    taylor='param_first',
    pruner_type='l2', 
    device=torch.device('cuda'),
    prune_by='block',
    layer=12,
    **kwargs
):

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    if pruner_type == 'random':
        imp = tp.importance.RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = llama_pruner.TaylorImportance(group_reduction=grouping_strategy, taylor=taylor)
    else:
        raise NotImplementedError
    
    logger.info("Use {} pruner...".format(pruner_type))

    forward_prompts = torch.tensor([
        [    1,   306,  4658,   278,  6593,   310,  2834,   338],
        [    1,  3439, 17632,  1925, 29892,   278,  6368,   310],
    ]).to(device)

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if prune_by == 'block':
        _prune_blockwise(
            model,
            tokenizer,
            imp,
            forward_prompts,
            logger,
            pruner_type=pruner_type,
            taylor=taylor,
            device=device,
            **kwargs
        )
    elif prune_by == 'channel':
        _prune_channelwise(
            model,
            tokenizer,
            imp,
            forward_prompts,
            logger,
            pruner_type=pruner_type,
            taylor=taylor,
            device=device,
            **kwargs
        )
    elif prune_by == 'layer':
        model.model.layers = model.model.layers[:layer]
    
    
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Clean the gradient in the model
    model.zero_grad()
    for name, module in model.named_parameters():
        if 'weight' in name:
            module.grad = None

    logger.info("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))

    gc.collect()
    torch.cuda.empty_cache()

    return model

'''
    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    #ddp
    parser.add_argument('--local_rank', type=int, default=-1)


'''


import transformers
from datasets import load_dataset

from src.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from src.utils.prompter import Prompter, ZeroPrompter
from src.datasets.ppl_dataset import get_loaders

def post_training(
    model,
    tokenizer,
    batch_size=128,
    micro_batch_size=4,
    device=torch.device("cuda"),
    no_instruction=False,
    prompt_template_name="alpaca",
    cutoff_len=256,
    data_path="yahma/alpaca-cleaned",
    train_on_inputs=False,
    add_eos_token=False,
    lora_r=8,
    lora_alpha=16,
    lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj",
    lora_dropout=0.05,
    cache_dataset=False,
    val_set_size=2000,
    extra_val_dataset=None,
    num_epochs=5,
    learning_rate=3e-4,
    output_dir="./lora-alpaca",
    group_by_length=False,
    resume_from_checkpoint=False
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    if not no_instruction:
        prompter = Prompter(prompt_template_name)
    else:
        prompter = ZeroPrompter()
    
    if device == 'cuda':
        model.half()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if 'lamini' in data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["response"],
            )
        elif 'alpaca' in data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
        else:
            raise NotImplementedError

        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set

    # Prepare For LoRA
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules.split(","),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()  

    # Load Train Dataset
    data = load_dataset(data_path)
    if cache_dataset and os.path.exists('datasets/cache/{}.bin'.format(data_path)):
        preprocess_data = torch.load('datasets/cache/{}.bin'.format(data_path))
        train_data, val_data = preprocess_data['train'], preprocess_data['val']
    else:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = {
            data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
        }
        if cache_dataset:
            cache_file = 'datasets/cache/{}.bin'.format(data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)

            torch.save({
                'train': train_data, 'val': val_data
            }, cache_file)

    # Load Extra Validation Dataset
    if extra_val_dataset:
        from src.datasets.ppl_dataset import get_wikitext2, get_ptb

        seq_len = 128
        for extra_dataset in extra_val_dataset.split(','):
            if 'wikitext2' in extra_dataset:
                _, test_data = get_wikitext2(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='text')
            if 'ptb' in extra_dataset:
                _, test_data = get_ptb(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='sentence')
            val_data[extra_dataset] = test_data
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=group_by_length,
            report_to="wandb",
            run_name=output_dir.split('/')[-1],
            metric_for_best_model="{}_loss".format(data_path),
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)