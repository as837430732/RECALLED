import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from multimodal_image_generator import MultimodalImagePerturbationGenerator
import random
import time
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import traceback
import copy

class MultimodalImageAttackTrainer:
    
    def __init__(self, model_path, model_type='qwen2vl', device='cuda', 
                 perturbation_type='pgd', steps=10, repeat_num=3, 
                 attack_type = 'token',
                 system_prompt="You are a helpful assistant."):
        
        self.model_type = model_type
        self.device = device
        self.perturbation_type = perturbation_type
        self.steps = steps
        self.repeat_num = repeat_num
        self.attack_type = attack_type
        self.system_prompt = system_prompt
        
        self.generator = MultimodalImagePerturbationGenerator(
            model_path=model_path,
            model_type=model_type,
            device=device,
            perturbation_type=perturbation_type,
            steps=steps,
            repeat_num=repeat_num,
            attack_type=attack_type
        )
        
        print(f"Multimodal image attack trainer initialized:")
        print(f"  Model type: {model_type}")
        print(f"  Perturbation type: {perturbation_type}")
        print(f"  Iteration steps: {steps}")

    def create_sample_image(self, width=224, height=224):
        from PIL import ImageDraw
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
        draw.ellipse([100, 100, 200, 200], fill='blue', outline='black')
        draw.text((10, 10), "Sample Image", fill='black')
        
        return img

    def prepare_data(self, orig_texts, labels, image_paths=None, file_names=None):

        if self.model_type.startswith('qwen2vl'):
            images = [Image.open(image_path).resize((224, 224)) for image_path in image_paths]
        elif self.model_type.startswith('blip2'):
            images = [Image.open(image_path).convert('RGB').resize((224, 224)) for image_path in image_paths]
        elif self.model_type.startswith('insblip'):
            images = [Image.open(image_path).convert('RGB').resize((224, 224)) for image_path in image_paths]
        else:
            images = [Image.open(image_path).resize((336, 336)) for image_path in image_paths]

        min_len = min(len(orig_texts), len(labels), len(images))
        orig_texts = orig_texts[:min_len]
        labels = labels[:min_len]
        images = images[:min_len]
        file_names = file_names[:min_len]

        return orig_texts, labels, images, file_names

    def train_adversarial_images(self, orig_texts, labels, image_paths=None, file_names=None, 
                                epochs=50, batch_size=4, log_path=None, save_dir='./adversarial_images/'):
        
        orig_texts, labels, images, file_names = self.prepare_data(orig_texts, labels, image_paths, file_names)
        
        if log_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_path = f'multimodal_image_attack_{self.model_type}_{self.perturbation_type}_log.txt'
        
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("=== Multimodal Image Perturbation Training Started ===\n")
            f.write(f"Perturbation type: {self.perturbation_type}\n")
            f.write(f"Iteration steps: {self.steps}\n\n")
        
        os.makedirs(save_dir, exist_ok=True)
        
        data = list(zip(orig_texts, labels, images, file_names))
        num_batches = len(data) // batch_size + int(len(data) % batch_size != 0)
        print(f"num_batches: {num_batches}")
        best_success_rate = 0
        best_adversarial_pixel_values = None
        best_original_inputs = None
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            epoch_success_count = 0
            epoch_total_count = 0
            
            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}"):
                batch = data[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_orig = [x[0] for x in batch]
                batch_labels = [x[1] for x in batch]
                batch_images = [x[2] for x in batch]
                batch_file_names = [x[3] for x in batch]
                try:
                  
                    adversarial_pixel_values, perturbations = self.generator.generate_batch_adversarial_pixel_values(
                        images=batch_images,
                        texts=batch_orig,
                        target_labels=batch_labels,
                        file_names=batch_file_names,
                        attack_type=self.perturbation_type,
                        targeted=True 
                    )                    
                  
                    original_inputs_list = self.generator.prepare_batch_multimodal_inputs(batch_images, batch_orig, target_labels=[])
                    
                    success_rate, responses = self.generator.evaluate_batch_adversarial_attack(
                        adversarial_pixel_values, original_inputs_list, perturbations, batch_orig, batch_labels
                    )                    
                    success_count = int(success_rate * len(batch_orig))
                    epoch_success_count += success_count
                    epoch_total_count += len(batch_orig)
                    
                    batch_success_rate = success_count / len(batch_orig)
                    self.record_batch_results(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        success_rate=batch_success_rate,
                        perturbation_stats=self.analyze_perturbation_stats(perturbations),
                        log_path=log_path
                    )
                    
                    print(f"Batch {batch_idx+1}: Success rate {batch_success_rate:.2%}")
                    if batch_success_rate >= best_success_rate:
                        best_success_rate = batch_success_rate
                       
                        self.save_batch_best_adversarial_results(
                            adversarial_pixel_values, 
                            batch_images,
                            batch_orig, 
                            save_dir, 
                            epoch, 
                            batch_idx,
                            perturbations
                        )                
                except Exception as e:
                    print(f"Batch {batch_idx+1} training error: {e}")
                    traceback.print_exc()
                    continue
            
            epoch_success_rate = epoch_success_count / epoch_total_count if epoch_total_count > 0 else 0
            print(f"Epoch {epoch+1} total success rate: {epoch_success_rate:.2%}")
            
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"\nEpoch {epoch+1} Summary:\n")
                f.write(f"  Total success rate: {epoch_success_rate:.2%}\n")
                f.write(f"  Best success rate: {best_success_rate:.2%}\n\n")
        
        
        self.save_attack_config(log_path.replace('.txt', '_config.pt'))
        
        print(f"\nTraining completed! Best attack success rate: {best_success_rate:.2%}")
        return best_adversarial_pixel_values, perturbations,best_success_rate

    def evaluate_attack_success(self, adversarial_pixel_values, original_inputs, orig_texts, labels):

        success_rate, responses = self.generator.evaluate_batch_adversarial_attack(
            adversarial_pixel_values, original_inputs, orig_texts, labels
        )
        
        success_count = int(success_rate * len(orig_texts))
        return success_count

    def analyze_perturbation_stats(self, perturbations):
        if perturbations is None:
            return {}
        
        stats = {
            'l2_norm': torch.norm(perturbations.view(perturbations.shape[0], -1), p=2, dim=1).mean().item(),
            'linf_norm': torch.norm(perturbations.view(perturbations.shape[0], -1), p=float('inf'), dim=1).mean().item(),
            'mean': perturbations.mean().item(),
            'std': perturbations.std().item(),
            'max': perturbations.max().item(),
            'min': perturbations.min().item()
        }
        
        return stats

    def record_batch_results(self, epoch, batch_idx, success_rate, perturbation_stats, log_path):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] Epoch {epoch+1}, Batch {batch_idx+1}\n")
            f.write(f"Attack success rate: {success_rate:.2%}\n")
            
            if perturbation_stats:
                f.write("Perturbation statistics:\n")
                for key, value in perturbation_stats.items():
                    f.write(f"  {key}: {value:.6f}\n")
            
            f.write("-" * 60 + "\n")

    def save_best_adversarial_results(self, adversarial_pixel_values, original_images, texts, save_dir, epoch, batch_idx):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        result_dir = os.path.join(save_dir, f'best_epoch{epoch+1}_batch{batch_idx+1}_{timestamp}')
        saved_paths = self.generator.save_adversarial_results(
            adversarial_pixel_values, original_images, texts, result_dir
        )
        
        print(f"Best results saved to: {result_dir}")
    
    def save_batch_best_adversarial_results(self, adversarial_pixel_values, original_images, texts, save_dir, epoch, batch_idx, perturbations):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        result_dir = os.path.join(save_dir, f'best_epoch{epoch+1}_batch{batch_idx+1}_{timestamp}')
        saved_paths = self.generator.save_batch_adversarial_results(
            adversarial_pixel_values, original_images, texts, perturbations, result_dir
        )
        
        print(f"Best results saved to: {result_dir}")
    
    def detailed_validation(self, adversarial_pixel_values, original_inputs, orig_texts, labels):
        print("\n=== Detailed Attack Effectiveness Validation ===")
        
        try:
            success_rate, responses = self.generator.evaluate_batch_adversarial_attack(
                adversarial_pixel_values, original_inputs, orig_texts, labels
            )
            
            print(f"Detailed validation completed, success rate: {success_rate:.2%}")
            
        except Exception as e:
            print(f"  Validation error: {e}")
            traceback.print_exc()

    def save_attack_config(self, config_path):
        config = {
            'model_type': self.model_type,
            'perturbation_type': self.perturbation_type,
            'steps': self.steps,
            'system_prompt': self.system_prompt
        }
        
        torch.save(config, config_path)
        print(f"Attack configuration saved to: {config_path}")

    def load_attack_config(self, config_path):
        config = torch.load(config_path, map_location=self.device)
        
        self.model_type = config['model_type']
        self.perturbation_type = config['perturbation_type']
        self.steps = config['steps']
        self.system_prompt = config['system_prompt']
        
        print(f"Attack configuration loaded from {config_path}")

    def visualize_attack_results(self, adversarial_pixel_values, original_images, perturbations, texts, save_path='./attack_visualization.png'):
        adversarial_images = self.generator.batch_pixel_values_to_images(adversarial_pixel_values, original_images, perturbations)
        
        num_samples = min(3, len(original_images))
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_samples):
            axes[0, i].imshow(original_images[i])
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(adversarial_images[i])
            axes[1, i].set_title(f'Adversarial Image {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Attack results visualization saved to: {save_path}") 