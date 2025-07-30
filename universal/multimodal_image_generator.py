import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration, Qwen2VLImageProcessor, AutoTokenizer, AutoProcessor,
    InstructBlipProcessor, InstructBlipForConditionalGeneration
)
import torchvision.transforms as transforms
from PIL import Image, ImageChops
import numpy as np
import copy 
import csv
import os
import matplotlib.pyplot as plt
from typing import Tuple, List
import math

class MultimodalImagePerturbationGenerator(nn.Module):
    
    def __init__(self, model_path=None, model=None, processor=None, model_type='qwen2vl', 
                 device='cuda', perturbation_type='pgd', steps=10,repeat_num=3,attack_type='token'):
        super().__init__()
        self.model_type = model_type
        self.device = device
        self.perturbation_type = perturbation_type
        self.steps = steps      
        self.repeat_num = repeat_num 
        self.attack_type = attack_type 
        
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        elif model_path is not None:
            self.model, self.processor = self._load_model(model_path, model_type, device)
        else:
            raise ValueError("Must provide model_path or (model, processor) pair")
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"Multimodal image perturbation generator initialized:")
        print(f"  Model type: {model_type}")
        print(f"  Device: {device}")
        print(f"  Perturbation method: {perturbation_type}")

    def _load_model(self, model_path, model_type, device):
        if model_type.startswith('llava'):
            processor = AutoProcessor.from_pretrained(model_path)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map=device
            )
        elif model_type.startswith('qwen2vl'):
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=device
            )
            processor = AutoProcessor.from_pretrained(model_path)
        elif model_type.startswith('insblip'):
            model = InstructBlipForConditionalGeneration.from_pretrained(
                model_path,
                device_map=device
            )
            processor = InstructBlipProcessor.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model, processor

    def prepare_multimodal_inputs(self, images, texts, target_labels):
        if self.model_type == 'llava':
            conversations = []
            for text in texts:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text}
                        ]
                    }
                ]
                conversations.append(conversation)
            
            inputs = self.processor(
                text=conversations,
                images=images,
                return_tensors="pt",
                padding=True
            )
            
        elif self.model_type == 'qwen2vl':
            messages_list = []
            for text in texts:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text}
                        ]
                    }
                ]
                messages_list.append(messages)
          
            text_prompt = self.processor.apply_chat_template(messages_list, add_generation_prompt=True)
          
            inputs = self.processor(
                text=text_prompt,
                images=images,
                return_tensors="pt",
                padding=True
            )
                
        return inputs

    def prepare_batch_multimodal_inputs(self, images, texts, target_labels):
        inputs_list = []
        if self.model_type.startswith('llava'):
            messages_list = []
            for text, image in zip(texts, images):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {"type": "image"}
                        ]
                    }
                ]
                text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(
                    text=text_prompt,
                    images=image,
                    return_tensors="pt",
                )
                inputs_list.append(inputs)
        
        elif self.model_type.startswith('insblip'):
            for text, image in zip(texts, images):
                inputs = self.processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                )     
                inputs_list.append(inputs)
            
        elif self.model_type.startswith('qwen2vl'):
            for text, image in zip(texts, images):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text}
                        ]
                    }
                ]
                text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(
                    text=text_prompt,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                inputs_list.append(inputs)
        return inputs_list

    def pgd_batch_attack(self, images, texts, target_labels, file_names, targeted=False):
        self.model.eval()
        print(f"Starting PGD attack, {self.steps} iterations...")
        inputs_list = self.prepare_batch_multimodal_inputs(images, texts, target_labels)
        for i in range(len(inputs_list)):
            inputs = inputs_list[i]
            if 'pixel_values' in inputs:
                original_pixel_values = inputs['pixel_values'].to(self.device)
            else:
                raise ValueError("pixel_values not found in input, please check processor configuration")
            
            print(f"Original pixel_values shape: {original_pixel_values.shape}")
            print(f"Value range: [{original_pixel_values.min():.4f}, {original_pixel_values.max():.4f}]")
    
            delta = torch.zeros_like(original_pixel_values, requires_grad=True, device=self.device)
            optimizer = torch.optim.Adam([delta], lr=1e-2)
            target_label = target_labels[i]
            tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            target_ids = tokenizer.encode(target_label, add_special_tokens=False, return_tensors='pt').to(self.device)
            inputs = inputs.to(self.device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], target_ids], dim=1)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(target_ids)], dim=1)

        log_list = []
        
        for step in range(self.steps):
            total_loss = 0
            for inputs in inputs_list:
                if delta.grad is not None:
                    delta.grad.zero_()
            

                perturbed_inputs = copy.deepcopy(inputs)
                original_pixel_values = inputs['pixel_values']
                perturbed_pixel_values = original_pixel_values + delta
                perturbed_inputs['pixel_values'] = perturbed_pixel_values

                
                for key in perturbed_inputs:
                    if isinstance(perturbed_inputs[key], torch.Tensor):
                        perturbed_inputs[key] = perturbed_inputs[key].to(self.device)
                
                outputs = self.model(**perturbed_inputs)
                logits = outputs.logits
                
                loss = self._compute_attack_loss(logits, target_labels, perturbed_inputs, targeted)
                total_loss += loss
            
            loss = total_loss / len(inputs_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            delta.data = torch.clamp(delta.data, -0.2, 0.2)
                
            if step % 5 == 0 or step == self.steps - 1:
                delta_linf = delta.abs().max().item()
                delta_l2 = delta.norm().item()
                print(f"  Step {step+1}/{self.steps}, Loss: {loss.item():.4f}, "
                      f"Perturbation L∞: {delta_linf:.6f}, Perturbation L2: {delta_l2:.6f}")
                if loss.item() < 0.1:
                    break
            log_list.append({
                'step': step+1,
                'loss': loss.item(),
                'delta_linf': delta_linf,
                'delta_l2': delta_l2
            })
        
        adversarial_pixel_values = original_pixel_values + delta
        print(adversarial_pixel_values)
        print(f"PGD attack completed! Final perturbation L∞ norm: {delta.abs().max().item():.6f}")
        print(f"Original L∞ norm: {original_pixel_values.abs().max().item():.6f}")
        class_names = file_names[0].split('_')
        save_dir = './pixel_values/'+ self.model_type + '/' + str(self.repeat_num) + '/' + self.attack_type 
        os.makedirs(save_dir, exist_ok=True)
        adversarial_pixel_values_path = os.path.join(save_dir, class_names[0]+'_tensor.pt')
        torch.save(adversarial_pixel_values, adversarial_pixel_values_path)
    
        adversarial_pixel_values_list = []
        for i in range(len(inputs_list)):
            inputs = inputs_list[i]
            original_pixel_values = inputs['pixel_values']
            adversarial_pixel_values = original_pixel_values + delta
            adversarial_pixel_values_list.append(adversarial_pixel_values)
        self.save_and_plot_training_log(log_list, save_dir='./log/'+ self.model_type + '/'+ str(self.repeat_num) +  '/' + self.attack_type + '/' +file_names[0])
        return adversarial_pixel_values_list, delta.detach()
       
   


    def _compute_attack_loss(self, logits, target_labels, inputs, targeted=False):
       
        target_labels = target_labels[0]
       
        tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        target_ids = tokenizer.encode(target_labels, add_special_tokens=False, return_tensors='pt').to(self.device)
      
        target_logits = logits[:, -target_ids.shape[1]-1:-1, :]
        target_logits = target_logits.reshape(-1, target_logits.shape[-1])
        target_ids_flat = target_ids.reshape(-1)       
       
        loss = F.cross_entropy(target_logits, target_ids_flat)
        
        
        return loss

    
    def generate_batch_adversarial_pixel_values(self, images, texts, target_labels, file_names, attack_type=None, targeted=False):
        attack_type = attack_type or self.perturbation_type
        
        if attack_type == 'pgd':
           
            return self.pgd_batch_attack(images, texts, target_labels, file_names, targeted)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")
    

    def batch_pixel_values_to_images(self, pixel_values, original_images, delta, original_shapes=None):
     
        images = []
      

        if self.model_type.startswith('qwen2vl'):
            images = self._batch_pixel_values_to_images_qwen2vl(pixel_values, original_images, delta, original_shapes)
        elif self.model_type.startswith('llava'):
            images = self._batch_pixel_values_to_images_llava(pixel_values, original_images, original_shapes)
        elif self.model_type.startswith('insblip'):
            images = self._batch_pixel_values_to_images_insblip(pixel_values, original_images,original_shapes)
        else:     
            pass        
        return images


    def _batch_pixel_values_to_images_qwen2vl(self, pixel_values, original_images, delta, original_shapes=None):
      
        images = []

        patch_size = getattr(self.processor.image_processor, 'patch_size', 14)
        temporal_patch_size = getattr(self.processor.image_processor, 'temporal_patch_size', 2)
        merge_size = getattr(self.processor.image_processor, 'merge_size', 2)
        image_mean = getattr(self.processor.image_processor, 'image_mean', [0.48145466, 0.4578275, 0.40821073])
        image_std = getattr(self.processor.image_processor, 'image_std', [0.26862954, 0.26130258, 0.27577711])
        rescale_factor = getattr(self.processor.image_processor, 'rescale_factor', 1/255)
        
        print(f"Qwen2VL parameters: patch_size={patch_size}, temporal_patch_size={temporal_patch_size}, merge_size={merge_size}")
        
       
        for original_image in original_images:
           
            original_image = original_image.resize((224, 224))
            original_width, original_height = original_image.size
            print(f"Original image size: {original_width}x{original_height}")
            
            image_processor = Qwen2VLImageProcessor()
            
            flatten_patches, grid_thw = image_processor._preprocess(
                images=original_image,
                do_resize=True,
                size={"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280},
                resample=image_processor.resample,
                do_rescale=True,
                rescale_factor=image_processor.rescale_factor,
                do_normalize=True,
                image_mean=image_processor.image_mean,
                image_std=image_processor.image_std,
                patch_size=image_processor.patch_size,
                temporal_patch_size=image_processor.temporal_patch_size,
                merge_size=image_processor.merge_size,
                do_convert_rgb=True
            )
            flatten_patches = torch.tensor(flatten_patches).to(self.device)
           
            img_data = flatten_patches + delta
            img_data = img_data.cpu().numpy()
          
            reconstructed_image = self.inverse_preprocess(
                        flatten_patches=img_data,
                        grid_thw=grid_thw,
                        original_height=original_height,
                        original_width=original_width,
                        patch_size=patch_size,
                        temporal_patch_size=temporal_patch_size,
                        merge_size=merge_size,
                        add_noise=False,
                        noise_scale=0
                    )

            images.append(reconstructed_image)
            
        return images
    
    def inverse_preprocess(self,
        flatten_patches: np.ndarray,
        grid_thw: Tuple[int, int, int],
        original_height: int,
        original_width: int,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        channel: int = 3,
        image_mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
        image_std: List[float] = [0.26862954, 0.26130258, 0.27577711],
        rescale_factor: float = 1/255,
        add_noise: bool = False,
        noise_scale: float = 0.0
    ):
    
        grid_t, grid_h, grid_w = grid_thw
   
    
        patches = flatten_patches.reshape(
            grid_t * grid_h * grid_w, 
            channel * temporal_patch_size * patch_size * patch_size
        )
        
        patches = patches.reshape(
            grid_t, grid_h, grid_w, 
            channel * temporal_patch_size * patch_size * patch_size
        )
        
        patches_9d_after_transpose = patches.reshape(
            grid_t, 
            grid_h // merge_size, 
            grid_w // merge_size, 
            merge_size, 
            merge_size, 
            channel, 
            temporal_patch_size, 
            patch_size, 
            patch_size
        )
     
        inverse_transpose_indices = [0, 6, 5, 1, 3, 7, 2, 4, 8]
        patches_9d_before_transpose = patches_9d_after_transpose.transpose(inverse_transpose_indices)
          
    
        patches_image_format = patches_9d_before_transpose.reshape(
            grid_t * temporal_patch_size,
            channel,
            grid_h // merge_size * merge_size * patch_size,
            grid_w // merge_size * merge_size * patch_size
        )
        
        patches_image_format = patches_9d_before_transpose.reshape(
            grid_t * temporal_patch_size,
            channel,
            grid_h * patch_size,
            grid_w * patch_size
        )
        
        
        if patches_image_format.shape[0] > 1:
            image = patches_image_format[0]
        else:
            image = patches_image_format[0]
        
      
        if add_noise and noise_scale > 0:
            noise = np.random.normal(0, noise_scale, image.shape)
            image = image + noise
            image = np.clip(image, -10, 10)
        
    
        image_mean = np.array(image_mean).reshape(3, 1, 1)
        image_std = np.array(image_std).reshape(3, 1, 1)
        image = image * image_std + image_mean

        image = image / rescale_factor
        
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        
      
        
        image = np.transpose(image, (1, 2, 0))
        
        pil_image = Image.fromarray(image)
        reconstructed_height, reconstructed_width = image.shape[:2]
        
        
        if (reconstructed_height, reconstructed_width) != (original_height, original_width):
            pil_image = pil_image.resize((original_width, original_height), Image.BICUBIC)
        
        return pil_image




    def _batch_pixel_values_to_images_llava(self, pixel_values, original_images, image_mean=None, image_std=None):
      
        images = []
        for pixel_value in pixel_values:
            if image_mean is None:
                image_mean = [0.48145466, 0.4578275, 0.40821073]
            if image_std is None:
                image_std = [0.26862954, 0.26130258, 0.27577711]
            
         
            if isinstance(pixel_value, torch.Tensor):
                pixel_value = pixel_value.detach().cpu().numpy()
            
            if len(pixel_value.shape) == 4:
                pixel_value = pixel_value[0]
            
            for c in range(3):
                pixel_value[c] = pixel_value[c] * image_std[c] + image_mean[c]
            
            pixel_value = pixel_value * 255
            
            pixel_value = np.transpose(pixel_value, (1, 2, 0))
            
            pixel_value = np.clip(pixel_value, 0, 255).astype(np.uint8)
            
            image = Image.fromarray(pixel_value)
            images.append(image)
        return images

   
    def _batch_pixel_values_to_images_insblip(self, pixel_values, original_images, delta, original_shapes=None):
        images = []
        for pixel_value in pixel_values:
            if isinstance(pixel_value, torch.Tensor):
                pixel_value = pixel_value.detach().cpu()
            if isinstance(delta, torch.Tensor):
                delta = delta.detach().cpu()
        
            pixel_value = pixel_value.squeeze(0)
            image_mean = [0.48145466, 0.4578275, 0.40821073]
            image_std = [0.26862954, 0.26130258, 0.27577711]
            
            mean = torch.tensor(image_mean).view(3, 1, 1)
            std = torch.tensor(image_std).view(3, 1, 1)
        
            img_tensor = pixel_value
            delta_tensor = delta
            
            perturbed_tensor = img_tensor
            
            perturbed_tensor = perturbed_tensor * std + mean
            
            perturbed_tensor = perturbed_tensor * 255
            
            perturbed_tensor = torch.clamp(perturbed_tensor, 0, 255)
            
            perturbed_tensor = perturbed_tensor.to(torch.uint8)
            
            transform = transforms.ToPILImage()
            image = transform(perturbed_tensor)
            
            
            images.append(image)
        
        return images

    def evaluate_batch_adversarial_attack(self, adversarial_pixel_values, original_inputs_list, perturbations, texts, target_labels):
        max_new_tokens = 500
        print("Evaluating adversarial attack effectiveness...")
        responses = []
        responses_len = []
        for original_inputs in (original_inputs_list):
            adversarial_inputs = copy.deepcopy(original_inputs)
            perturbations.to(self.device)
            adversarial_inputs.to(self.device)
            adversarial_inputs['pixel_values'] += perturbations
            
            for key in adversarial_inputs:
                if isinstance(adversarial_inputs[key], torch.Tensor):
                    adversarial_inputs[key] = adversarial_inputs[key].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **adversarial_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    return_dict_in_generate=True,
                )
            
            if 'input_ids' in adversarial_inputs:
                input_length = adversarial_inputs['input_ids'].shape[1]
            else:
                input_length = 0
            
            output_ids = outputs.sequences[:, input_length:]
            tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            
            
            for i in range(output_ids.shape[0]):
                response = tokenizer.decode(output_ids[i], skip_special_tokens=True)
            responses.append(response)
            responses_len.append(len(output_ids[i]))
        
        success_count = 0
     
        for i, response_len in enumerate(responses_len):
            is_success = (response_len == max_new_tokens)
            if is_success:
                success_count += 1    

            print(f"Sample {i+1}:")
            print(f"  Text: {texts[i]}",flush=True)
            print(f"  Target: {target_labels[i] if isinstance(target_labels, list) else target_labels}",flush=True)
            print(f"  Response: {responses[i][:int(len(responses[i])/2)]}", flush=True)
            print(f"  Success: {'Yes' if is_success else 'No'}",flush=True)

        success_rate = success_count / len(responses)
        print(f"Overall attack success rate: {success_rate:.2%}",flush=True)
        import sys
        sys.stdout.flush()
        
        return success_rate, responses


    def save_adversarial_results(self, adversarial_pixel_values, original_images, texts, 
                                save_dir='./adversarial_results/'):
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        adversarial_images = self.pixel_values_to_images(adversarial_pixel_values)
        
        saved_paths = []
        for i, (orig_img, adv_img, text) in enumerate(zip(original_images, adversarial_images, texts)):
            orig_path = os.path.join(save_dir, f'original_{i}.png')
            orig_img.save(orig_path)
            
            adv_path = os.path.join(save_dir, f'adversarial_{i}.png')
            adv_img.save(adv_path)
            
            text_path = os.path.join(save_dir, f'info_{i}.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Original text: {text}\n")
                f.write(f"Attack method: {self.perturbation_type}\n")
                f.write(f"Perturbation parameters: epsilon={self.epsilon:.4f}, alpha={self.alpha:.4f}, steps={self.steps}\n")
            
            saved_paths.extend([orig_path, adv_path, text_path])
        
        print(f"Adversarial attack results saved to: {save_dir}")
        return saved_paths

    def save_batch_adversarial_results(self, adversarial_pixel_values, original_images, texts, delta,
                                save_dir='./adversarial_results/'):
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        adversarial_images = self.batch_pixel_values_to_images(adversarial_pixel_values, original_images, delta)
        
        saved_paths = []
        for i, (orig_img, adv_img, text) in enumerate(zip(original_images, adversarial_images, texts)):
            orig_path = os.path.join(save_dir, f'original_{i}.png')
            orig_img.save(orig_path)
            
            adv_path = os.path.join(save_dir, f'adversarial_{i}.png')
            adv_img.save(adv_path)

            text_path = os.path.join(save_dir, f'info_{i}.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"Original text: {text}\n")
                f.write(f"Attack method: {self.perturbation_type}\n")

            
            saved_paths.extend([orig_path, adv_path, text_path])
        
        print(f"Adversarial attack results saved to: {save_dir}")
        return saved_paths

    def analyze_perturbation(self, original_pixel_values, adversarial_pixel_values, save_dir='./perturbation_analysis/'):
        import os
        import matplotlib.pyplot as plt
        
        os.makedirs(save_dir, exist_ok=True)
        
        perturbations = adversarial_pixel_values.cpu() - original_pixel_values.cpu()
        
        stats = {
            'l2_norm': torch.norm(perturbations.view(perturbations.shape[0], -1), p=2, dim=1).mean().item(),
            'linf_norm': torch.norm(perturbations.view(perturbations.shape[0], -1), p=float('inf'), dim=1).mean().item(),
            'mean_perturbation': perturbations.mean().item(),
            'std_perturbation': perturbations.std().item(),
            'max_perturbation': perturbations.max().item(),
            'min_perturbation': perturbations.min().item()
        }
        
        print("Perturbation statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.6f}")
        
        return stats

    def save_and_plot_training_log(self, log_list, save_dir='./training_log/'):
        import os
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'training_log.csv')
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_list[0].keys())
            writer.writeheader()
            writer.writerows(log_list)
        print(f"Training log saved to: {csv_path}")

        steps = [item['step'] for item in log_list]
        losses = [item['loss'] for item in log_list]
        linfs = [item['delta_linf'] for item in log_list]
        l2s = [item['delta_l2'] for item in log_list]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, label='Loss')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('Training Loss and Perturbation Norms')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'training_log_plot.png'))
        plt.show()
        print(f"Training process curve saved to: {os.path.join(save_dir, 'training_log_plot.png')}")

