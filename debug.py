import os
import json
from collections import OrderedDict
from typing import List, Dict, Any, Union, Tuple
import datasets
from datasets import load_dataset
from tqdm import tqdm
import copy
import fire
import torch
import numpy as np
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainerCallback
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from fed_utils.model_aggregation import FedAvg
from fed_utils.client_participation_scheduling import client_selection
from utils.prompter import Prompter
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(f'runs/Debug2')


# Intialize Set Up
device = "cuda" if torch.cuda.is_available() else "cpu"
datasets.utils.logging.set_verbosity_error()
Scalar = Union[bool, bytes, float, int, str]


class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, tokenizer, ddp):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.ddp = ddp
        # Set input & output paths
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))

    # Partition Local Dataset
    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    # Trainer Helper Functions
    def build_local_trainer(self,
                            local_micro_batch_size: int = 8,
                            gradient_accumulation_steps: int = 8,
                            local_num_epochs: int = 10,
                            local_learning_rate: float = 3e-4,
                            group_by_length: bool = False,
                            ):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            warmup_steps=0,
            fp16=False,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=200,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if self.ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                    self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                                                    ),
                                                  )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    # Aggregation Helper Function (not needed for flwr)
    def reset_parameters(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        last_client_id = self.client_id
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)

        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id

    # Mimic Flower Client
    class LossCaptureCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and 'loss' in logs:
                self.losses.append(logs['loss'])

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        new_adapter_weight = self.model.state_dict()
        parameters = [v.cpu().numpy() for v in new_adapter_weight.values()]
        return parameters
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        try:
            self.build_local_trainer()
            self.initiate_local_training()

            loss_capture_callback = self.LossCaptureCallback()
            self.local_trainer.add_callback(loss_capture_callback)
            self.local_trainer.train()

            # Debugging prints
            parameters = self.get_parameters({})
            print("Parameters:", parameters)
            print("Type of Parameters:", type(parameters))

            loss = {"Loss": loss_capture_callback.losses[-1]}
            print("Loss:", loss)
            print("Type of Loss:", type(loss))
            return parameters, len(self.local_train_dataset), loss
        except Exception as e:
            print(f"Error occurred: {type(e).__name__}, {e}")
    

def fl_finetune(
        # model/data params
        global_model: str = 'chavinlo/alpaca-native',
        data_path: str = './data',
        output_dir: str = './lora-shepherd-debug/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.2, # 0.1
        num_communication_rounds: int = 2, # 10
        num_clients: int = 5, # 10 
        # Local training hyperparams
        local_batch_size: int = 64,  
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 10,
        local_learning_rate: float = 3e-4,
        local_val_set_size: float = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["q_proj",],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    
    # Check if the model & data path exists
    assert global_model, "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"
    # data_path = os.path.join(data_path, str(num_clients))
    data_path = os.path.join(data_path, "10")
    assert os.path.exists(data_path), "Please generate the data files for each client"

    # Set up parallel training
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Load global model & toknizer
    model_path = './models--chavinlo--alpaca-native/snapshots/3bf09cbff2fbd92d7d88a0f70ba24fca372befdf'
    model = LlamaForCausalLM.from_pretrained(
        # global_model,
        model_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        local_files_only=True
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    # tokenizer = LlamaTokenizer.from_pretrained(global_model)
    tokenizer.pad_token_id = (0)
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
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # Prepare for Tuning
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Simulate Local Training
    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = os.path.join(output_dir, str(num_clients))

    for epoch in tqdm(range(num_communication_rounds)):
        print("\nConducting the client selection")
        selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy)
        for client_id in selected_clients_set:
            client = GeneralClient(client_id, model, data_path, output_dir, tokenizer, ddp)
            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.preprare_local_dataset(generate_and_tokenize_prompt, local_val_set_size)
            print("Training..")
            new_param = client.fit([], {})[0]

        # flwr has built-in methods for aggregation
        print("Aggregating weights from each client..")
        model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.reset_parameters(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
        del client
        model = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       )
        torch.save(model.state_dict(), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)
    
        # Global Evaluation
        model.eval()
        correct_predictions = 0
        test_cases_path = './data_leaf/testing/shakespeare_instruction_response_pairs_all.json'
        test_cases = load_dataset("json", data_files=test_cases_path)
        for case in tqdm(test_cases["train"]):
            # Generate prompt
            prompt = prompter.generate_prompt(case["instruction"], case["context"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            # Generate output
            with torch.no_grad():
                outputs = model.generate(inputs["input_ids"], max_new_tokens=1, num_return_sequences=1)
            # Decode generated ID to text
            predicted_char = tokenizer.decode(outputs[:,-1][0], skip_special_tokens=True)
            expected_char = case["response"].strip()
            # Evaluate prediction
            if predicted_char.lower() == expected_char.lower():
                correct_predictions += 1
        accuracy = correct_predictions / len(test_cases["train"])
        num_examples = len(test_cases["train"])
        print(f"Round {epoch} , Accuracy: {accuracy:.2f} ({correct_predictions}/{num_examples})\n")
        writer.add_scalar("Testing accuracy at server round", accuracy, epoch)
        
        model.train()

    writer.close()


if __name__ == "__main__":
    fire.Fire(fl_finetune)
