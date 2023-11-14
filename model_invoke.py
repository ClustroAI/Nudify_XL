from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")

pipe.load_lora_weights("Remilistrasza/NSFW_LoRAs", weight_name="nudify_xl.safetensors", revision="Nudify_XL")

def invoke(input_text):
    prompt = input_text
    # prompt = '''cinematic photo, highres, photorealistic,
    # beautiful korean woman, nude, large breasts,
    # black hair, smile, cute girl, eyelashes,
    # hand on hip, topless, choker, black thighhighs, socks,
    # bedroom, studio lighting, hard shadows, sunset,
    # <lora:nudify:1>'''
    negative_prompt = '''blurry, painting, drawing, cartoon, rendered, 
    cgi, 3d, anime, sketch, (worst quality:2), (low quality:2), 
    (normal quality:2), (bad quality:2), dot, mole, monochrome, 
    grayscale, text, error, cropped, jpeg artifacts, ugly, duplicate, 
    morbid, mutilated, out of frame, mutation, deformed, dehydrated, 
    bad anatomy, bad proportions, disfigured, username, watermark, signature, 
    mature, granny, elder, wrinkled skin, saggy, ugly, penis, boy, male, scrotum, 
    tattoo, censor, fingers,
    '''
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        guidance_scale=7.5,
        num_inference_steps=25,
        #generator=torch.manual_seed(3885156286),
        cross_attention_kwargs={"scale": 0.5}
    ).images[0]
    image.save("generated_image.png")
    return "generated_image.png"