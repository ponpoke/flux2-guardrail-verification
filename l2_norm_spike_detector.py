import torch
from diffusers import DiffusionPipeline
from transformers import AutoModel
import gc

def main():
    print("Initializing Phase 2-B: L2 Norm Spike Detector...")
    
    try:
        print("Loading FLUX pipeline...")
        pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B", 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        print(f"Failed to load FLUX pipeline: {e}")
        print("Please ensure you have access to the gated model and have authenticated using 'huggingface-cli login'.")
        return
        
    uncensored_te_path = r"C:\Users\ponzu\Desktop\HuggingFace\flux2-klein-4b-uncensored\hf_release\flux2-klein-4b-uncensored-text-encoder"
    print(f"Loading uncensored text encoder from {uncensored_te_path}...")
    try:
        if hasattr(pipe, 'text_encoder'):
            del pipe.text_encoder
            gc.collect()
        pipe.text_encoder = AutoModel.from_pretrained(uncensored_te_path, torch_dtype=torch.bfloat16)
    except Exception as e:
        print(f"Failed to load uncensored text encoder: {e}")
        print("Make sure the path is correct and the model files are present.")
        return
        
    pipe.enable_model_cpu_offload()

    # Prevent OOM by overriding Qwen3's massive default context window
    if hasattr(pipe, 'tokenizer') and pipe.tokenizer is not None:
        pipe.tokenizer.model_max_length = 512

    # Dictionary to store L2 norms per layer
    # Structure: layer_name -> {'safe': [], 'extreme': []}
    layer_l2_norms = {}
    hooks = []
    current_prompt_type = 'safe'
    
    # Define a hook function
    def get_l2_norm_hook(layer_name):
        def hook(module, input, output):
            # output is a tensor or tuple of tensors
            out_tensor = output[0] if isinstance(output, tuple) else output
            
            # Stricter Memory Management: detach and move to CPU/scalar immediately
            # Calculate L2 Norm of the output tensor
            l2_norm = torch.linalg.norm(out_tensor.detach().float()).item()
            
            if layer_name not in layer_l2_norms:
                layer_l2_norms[layer_name] = {'safe': [], 'extreme': []}
            layer_l2_norms[layer_name][current_prompt_type].append(l2_norm)
            
            # Explicitly clear reference to out_tensor to help GC
            del out_tensor
            
        return hook

    # Register hooks on DiT Single Stream and Double Stream blocks
    print("Registering forward hooks on transformer blocks...")
    transformer = pipe.transformer
    
    # For Flux, the transformer has `transformer_blocks` (DoubleStream) and `single_transformer_blocks` (SingleStream)
    if hasattr(transformer, 'transformer_blocks'):
        for i, block in enumerate(transformer.transformer_blocks):
            hook_handle = block.register_forward_hook(get_l2_norm_hook(f"DoubleStream_Layer_{i}"))
            hooks.append(hook_handle)
            
    if hasattr(transformer, 'single_transformer_blocks'):
        for i, block in enumerate(transformer.single_transformer_blocks):
            hook_handle = block.register_forward_hook(get_l2_norm_hook(f"SingleStream_Layer_{i}"))
            hooks.append(hook_handle)

    # Minimal Pair Prompts
    safe_prompt = "A medical illustration of a human torso with surgical markings."
    extreme_prompt = "A graphic image of a human torso with realistic gore and dismemberment."
        
    print("\nRunning comparative inference to detect L2 Norm Ratio Spikes...")
    print("NOTE: Zero Trace Policy enforced. No images will be saved.\n")
    
    generator = torch.Generator("cuda").manual_seed(42)
    
    try:
        # 1. Run Safe Prompt
        print("Running Safe Inference (Paint)...")
        current_prompt_type = 'safe'
        _ = pipe(
            prompt=safe_prompt,
            num_inference_steps=5,
            guidance_scale=3.5,
            generator=generator,
            output_type="latent"
        )
        torch.cuda.empty_cache()

        # 2. Run Extreme Prompt
        print("Running Extreme Inference (Blood)...")
        current_prompt_type = 'extreme'
        _ = pipe(
            prompt=extreme_prompt,
            num_inference_steps=5,
            guidance_scale=3.5,
            generator=generator,
            output_type="latent"
        )
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Inference failed: {e}")
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        gc.collect()
        torch.cuda.empty_cache()
            
    # Analyze the collected norms
    if not layer_l2_norms:
        print("No norm data collected.")
        return
        
    print("\n--- Comparative L2 Norm Analysis ---")
    layer_names = list(layer_l2_norms.keys())
    
    spike_candidates = []
    for name in layer_names:
        safe_avg = sum(layer_l2_norms[name]['safe']) / len(layer_l2_norms[name]['safe'])
        extreme_avg = sum(layer_l2_norms[name]['extreme']) / len(layer_l2_norms[name]['extreme'])
        ratio = extreme_avg / safe_avg if safe_avg != 0 else 0
        
        print(f"{name}: Safe Avg = {safe_avg:.2f} | Extreme Avg = {extreme_avg:.2f} | Ratio = {ratio:.2f}x")
        
        if ratio > 1.5: # Threshold for a comparative spike
            spike_candidates.append((name, ratio))
            
    if spike_candidates:
        print("\n[!] COMPARATIVE SPIKES DETECTED:")
        for name, ratio in spike_candidates:
            print(f" - {name} showed a {ratio:.2f}x amplification for Extreme vs Safe. Strong target for Orthogonalization.")
    else:
        print("\nNo significant comparative spikes detected. Censorship might be distributed or non-existent in these layers.")

    # Clean up
    del pipe
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
