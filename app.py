#imports
import gradio as gr

import os
import glob
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn

import decord
decord.bridge.set_bridge('torch')

from video_llama.tasks import *
from video_llama.models import *
from video_llama.runners import *
from video_llama.processors import *
from video_llama.datasets.builders import *
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def upload_imgorvideo(video_path, chat_state):
    chat_state = default_conversation.copy()
    chat_state = Conversation(
        system= "",
        roles=("Human", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="###",
    )
    img_list = []
    chat.upload_video(video_path, chat_state, img_list)
    return chat_state, img_list
 
def gradio_ask(user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state

def gradio_answer(chat_state, img_list, num_beams, temperature):
    output_text, _ = chat.answer(conv=chat_state,
                                 img_list=img_list,
                                 num_beams=num_beams,
                                 temperature=temperature,
                                 max_new_tokens=300, 
                                 max_length=2000) # llama: max_token_num=2048
    return output_text

def infer(video_path):
    print(f'\n\n-----------------------{video_path}----------------------\n\n')
    chat_state, img_list = [], []
    chat_state, img_list = upload_imgorvideo(video_path, chat_state)
    chat_state = gradio_ask(user_message, chat_state)
    response = gradio_answer(chat_state, img_list, num_beams=1, temperature=1)
    print (f'assistant: {user_message}')
    print (f'answer: {response}')
    
    if 'Yes' in response:
        return "The video is HATEFUL"
    elif 'No' in response:
        return "The video is NOT HATEFUL"
    else:
        return response

parser = argparse.ArgumentParser(description="Inference Process for Multimodal Hate Content Detection")

# default config
parser.add_argument("--gpu-id", type=int, default='0', help="specify the gpu to load the model.")
parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio_stage3.yaml', help="path to configuration file.")
parser.add_argument("--options", nargs="+", help="override some settings in the used config, format: --option xx=xx yy=yy zz=zz")

# input message
parser.add_argument('--user_message',type=str, default="Is this hateful? Answer (Yes/No)", help='input user message')

args = parser.parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
print(f'--------------MODEL CONFIG------------ :\n{model_config}\n\n ----------------------------------------------------------------\n\n')
if args.gpu_id == -1:
    device = 'cpu'
else:
    device='cuda:{}'.format(args.gpu_id)
print(f'\n\n------------------device == {device}-----------\n\n')
model_config.device_8bit = device
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

chat = Chat(model, vis_processor, device=device)
user_message = args.user_message
print('Initialization Finished')

print ('Step2: feed-forward process')
#title = "Hate-LLaMA - An Instruction-tuned Audio-Visual Language Model for Hate Content Detection"
 
description = """
<h1 align="center"> Hate-LLaMA </h1>
<h3 align="center">  An Audio-Visual Language Model for Hate Content Detection </h3>

Hate-LLaMA , is a multi-modal framework, designed to detect hate in videos and classify them as HATE or NON HATE. Hate-LLaMA finetunes Video-LLaMA (which uses the LLaMA-7b-chat model as backbone). The model is able to analyse both the audio and visual content to perform the classification task. After uploading a video and clicking submit, the model outputs a simple statement identifying if the video has hate or not. 

"""

article = "Authors : Anisha Bhatnagar, Simran Makariye, Divyanshi Parashar"
#examples = ["examples/hate_video_136.mp4","examples/hate_video_2.mp4", "examples/non_hate_video_349.mp4", "examples/non_hate_video_569.mp4"]

demo = gr.Interface(fn=infer, inputs="video", outputs="text", description=description, article=article) #, examples=examples)

demo.launch(share=True,show_api=False)   


