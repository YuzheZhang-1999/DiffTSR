import os
from PIL import Image
from omegaconf import OmegaConf

from model.IDM.utils.util import instantiate_from_config


if __name__ == '__main__':
    DiffTSR_yaml_config = './model/DiffTSR_config.yaml'
    DiffTSR_ckpt_config = './ckpt/DiffTSR.ckpt'
    DiffTSR_config = OmegaConf.load(DiffTSR_yaml_config)
    DiffTSR_model = instantiate_from_config(DiffTSR_config.model)
    DiffTSR_model.load_model(DiffTSR_ckpt_config)

    input_lr_path = './testset/0_lr_synth/'
    sr_save_path = './testset/0_sr_synth/'

    input_lr_file_list = sorted(os.listdir(input_lr_path))

    for file_name in input_lr_file_list:
        input_image = os.path.join(input_lr_path, file_name)
        save_path = os.path.join(sr_save_path, file_name)
        lq_image_pil = Image.open(input_image).convert('RGB')
        # Start sampling!
        sr_output = DiffTSR_model.DiffTSR_sample(lq_image_pil)
        # Save sr image!
        sr_image_pil = Image.fromarray(sr_output, 'RGB')
        sr_image_pil.save(save_path)
