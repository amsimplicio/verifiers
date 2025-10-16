import argparse

import verifiers as vf

"""
# install
vf-install wordle (-p /path/to/environments)

# quick eval
vf-eval wordle -m (model_name in endpoints.py)

1.7b inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-1.7B-Wordle \
    --data-parallel-size 6 --enforce-eager --disable-log-requests

1.7b training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 1.7B

4b inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 vf-vllm --model willcb/Qwen3-4B-Wordle \
    --data-parallel-size 6 --enforce-eager --disable-log-requests

4b training:
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num-processes 2 \
    --config-file configs/zero3.yaml examples/grpo/train_wordle.py --size 4B
"""


def main(args):
    import os
    import torch
    
    # Ensure we're using the correct device
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")
        if visible_devices == '1':
            device = torch.device('cuda:0')  # This will be GPU 1 in the original numbering
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cuda:0')
    
    print(f"Using device: {device}")
    torch.cuda.set_device(device)
    
    size = args.size
    model_name = f"/gpfs/scratch/epor32/amsimplicio/rlvr/outputs/47-test/checkpoint-48_merged"
    model, tokenizer = vf.get_model_and_tokenizer(model_name)
    vf_env = vf.load_environment(env_id="wordle", use_think=True)
    run_name = f"wordle-grpo-{size}"
    training_args = vf.grpo_defaults(run_name=run_name)
    
    # Set the VLLM server port if provided
    if hasattr(args, 'vllm_port') and args.vllm_port:
        training_args.vllm_server_port = args.vllm_port
    training_args.per_device_train_batch_size = 1
    training_args.num_generations = 2  # Reduce to minimum for memory
    training_args.gradient_accumulation_steps = 8  # Increase to maintain effective batch size
    training_args.max_tokens = 256  # Reduce further to save memory
    training_args.max_seq_len = 1024  # Reduce sequence length further
    training_args.max_steps = 200
    training_args.gradient_checkpointing = True
    training_args.dataloader_pin_memory = False
    training_args.dataloader_num_workers = 0  # Reduce memory usage
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 20
    training_args.mask_env_responses = True
    training_args.max_grad_norm = 0.1
    training_args.beta = 0.0
    lora = vf.lora_defaults(r=1, alpha=16)  # Smaller rank for less memory

    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        lora_config=lora,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-s", type=str, default="1.7B")
    parser.add_argument("--vllm-port", type=int, help="VLLM server port")
    args = parser.parse_args()
    main(args)
