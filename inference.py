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

def upload_imgorvideo(video_path, chat_state, subtitle=None):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Process for Multimodal Hate Content Detection")

    # default config
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--cfg-path", default='eval_configs/video_llama_eval_withaudio_stage3.yaml', help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config, format: --option xx=xx yy=yy zz=zz")

    # input for emotion reasoning dataset
    parser.add_argument('--video_root',  type=str, default="/scratch/ab10945/LLVM/Correct_hatemm/test_videos_new",  help='video root')
    parser.add_argument('--label_path',  type=str, default="/scratch/ab10945/LLVM/Correct_hatemm/test_new.json",  help='label path')
    parser.add_argument('--no_subtitle', action='store_true', default=False, help='whether use subtitle in the inference (A+V+T)')
    parser.add_argument('--user_message',type=str, default="A hateful video is a video that promotes animosity, hostility, or violence against individuals or groups based on attributes such as their race, ethnicity, religion, national origin, gender, sexual orientation, disability, or other characteristic. It contains Derogatory Language, like racial slurs. Analyze both the audio and video content. Is this hateful? Provide your answer as a single word, either ‘Yes’ or ‘No’, without any accompanying explanation.", help='input user message')

    # test multiple saved files (accelerate)
    parser.add_argument('--ckpt_root',    type=str, default=None,  help='test multiple files')
    parser.add_argument('--test_epochs',  type=str, default=None,  help='test which epochs')
    args = parser.parse_args()
    cfg = Config(args)
    
    model_config = cfg.model_cfg
    print(f'--------------MODEL CONFIG------------ :\n{model_config}\n\n ----------------------------------------------------------------\n\n')
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')

    print ('Step2: feed-forward process')
    whole_results = {}
    df = pd.read_json(args.label_path)
    for _, row in df.iterrows():
        name     = row['video']
#             emotion  = row['QA']
#             subtitle = row['subtitles']
        answer   = row['QA'][0]['a']
        print (f'process on {name}')
        video_path = os.path.join(args.video_root, f'{name}')
        user_message = args.user_message

        # process for one file
        chat_state, img_list = [], []
        subtitle = None
        chat_state, img_list = upload_imgorvideo(video_path, chat_state, subtitle=subtitle)
        chat_state = gradio_ask(user_message, chat_state)
        response = gradio_answer(chat_state, img_list, num_beams=1, temperature=1)
        print (f'assistant: {user_message}')
        print (f'answer: {response}')
        whole_results[name] = {
            'answer': answer,
            'pred_answer': response,
        }


    print ('Step3: save results for one ckpt_3')
    save_path = os.path.join('ckpt5.npz')
    np.savez_compressed(save_path,whole_results=whole_results)
