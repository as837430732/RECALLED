import os
import torch
import csv
import argparse
from PIL import Image, ImageDraw
from multimodal_image_trainer import MultimodalImageAttackTrainer
import warnings
import time
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument('--model_type', type=str, default='qwen2vl3b', help='model type')
parser.add_argument('--attack_type', type=str, default='token', help='attack type')
parser.add_argument('--repeat_num', type=int, default='3', help='repeat num')
parser.add_argument('--steps', type=int, default='1000', help='steps')
parser.add_argument('--batch_size', type=int, default='5', help='batch_size')

args = parser.parse_args()

MODEL_TYPE = args.model_type  
ATTACK_TYPE = args.attack_type 
REPEAT_NUM = args.repeat_num 
STEPS = args.steps 
BATCH_SIZE = args.batch_size 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PERTURBATION_TYPE = 'pgd'   
EPOCHS = 1      
SYSTEM_PROMPT = "You are a helpful assistant."

MODEL_PATHS = {
    'llava7b': './models/llava-1.5-7b-hf',
    'llava13b': './models/llava-1.5-13b-hf',
    'insblip7b': './models/instructblip-vicuna-7b',
    'insblip13b': './models/instructblip-vicuna-13b',
    'qwen2vl3b': './models/Qwen2.5-VL-3B-Instruct',
    'qwen2vl7b': './models/Qwen2.5-VL-7B-Instruct',
    'qwen2vl32b': './models/Qwen2.5-VL-32B-Instruct',
}

def load_repeat_behaviors_data(file_path, max_samples=20):
    orig_texts = []
    sentence_labels = []
    token_labels = []
    image_paths = []
    file_names = []
    requests = []
    token_repeat_3 = []
    sentence_repeat_3 = []
    token_repeat_5 = []
    sentence_repeat_5 = []
    token_repeat_10 = []
    sentence_repeat_10 = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            orig_texts.append(row['original_text'])
            image_paths.append(row['path'])
            file_names.append(row['filename'])
            requests.append(row['Request'])
            token_repeat_3.append(row['token_repeat_3'])
            sentence_repeat_3.append(row['sentence_repeat_3'])
            token_repeat_5.append(row['token_repeat_5'])
            sentence_repeat_5.append(row['sentence_repeat_5'])
            token_repeat_10.append(row['token_repeat_10'])
            sentence_repeat_10.append(row['sentence_repeat_10'])

    return orig_texts,image_paths,file_names,requests,token_repeat_3, \
    sentence_repeat_3,token_repeat_5,sentence_repeat_5,token_repeat_10,sentence_repeat_10


def main():
    print("=== Multimodal Image Adversarial Attack System ===")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Model path: {MODEL_PATHS[MODEL_TYPE]}")
    print(f"Perturbation type: {PERTURBATION_TYPE}")
    print(f"Training parameters: epochs={EPOCHS}, batch_size={BATCH_SIZE}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Warning: CUDA not available, will use CPU (slow)")
    
    print("\nLoading training data...")
   
    if MODEL_TYPE.startswith('qwen2vl'):
        data_file_path = '../data/repeat_responses_with_repeats_qwen.csv'
    elif MODEL_TYPE.startswith('llava'):
        data_file_path = '../data/repeat_responses_with_repeats_llava.csv'
    elif MODEL_TYPE.startswith('insblip'):
        data_file_path = '../data/repeat_responses_with_repeats_blip.csv'
    else:
        print('No suitable dataset found')    
    orig_texts,image_paths,file_names,requests,token_repeat_3, \
    sentence_repeat_3,token_repeat_5,sentence_repeat_5,token_repeat_10,sentence_repeat_10 = load_repeat_behaviors_data(data_file_path, max_samples=5)


    if ATTACK_TYPE == 'token':
        if REPEAT_NUM == 3:
            labels = token_repeat_3
        elif REPEAT_NUM == 5:
            labels = token_repeat_5
        else:
            labels = token_repeat_10
    else:
        if REPEAT_NUM == 3:
            labels = sentence_repeat_3
        elif REPEAT_NUM == 5:
            labels = sentence_repeat_5
        else:
            labels = sentence_repeat_10

    print(f"Loaded {len(orig_texts)} text samples")
    print("Sample data:")
    for i in range(min(3, len(orig_texts))):
        print(f"  {i+1}. Original text: {orig_texts[i]}")
        print(f"     Target label: {labels[i]}")
    
    
    print(f"\nInitializing attack trainer...")
    trainer = MultimodalImageAttackTrainer(
        model_path=MODEL_PATHS[MODEL_TYPE],
        model_type=MODEL_TYPE,
        device=DEVICE,
        perturbation_type=PERTURBATION_TYPE,
        steps=STEPS,
        repeat_num=REPEAT_NUM,
        attack_type=ATTACK_TYPE,
        system_prompt=SYSTEM_PROMPT
    )
    
    print(f"\nStarting image adversarial attack training...")
    print("=" * 60)
    
    try:
        best_adversarial_pixel_values, perturbations, best_success_rate = trainer.train_adversarial_images(
            orig_texts=orig_texts,
            labels=labels,
            image_paths=image_paths,
            file_names=file_names,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            log_path=f'./log/{MODEL_TYPE}/image_attack_{PERTURBATION_TYPE}_log.txt',
            save_dir=f'./adversarial_images/{MODEL_TYPE}/'
        )
        
        print("=" * 60)
        print("Training completed!")
        print(f"Best attack success rate: {best_success_rate:.2%}")
        
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nProgram execution completed!")


if __name__ == "__main__":
      
    main() 
      