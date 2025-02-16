import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

HEIGHT = 512
WIDTH = 512
LATENT_HEIGHT = HEIGHT // 8 
LATENT_WIDTH = WIDTH // 8


def generate(prompt: str, uncond_prompt: str, input_image=None,
                strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name='ddpm',
                n_inference_steps=50, models={}, seed=None, device=None,
                idle_device=None,
                tokenizer=None):
    
    with torch.no_grad():

        if not (0 < strength <= 1):
            raise ValueError("strength needs to be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models['clip']
        clip = clip.to(device)

        # we go to the unet 2 times, one with the cond prompt and another one without
        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            # (batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding='max_length', max_length=77).input_ids
            uncond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (batch_size, seq_len) -> (batch_size, seq_len, dim)
            uncond_context  = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context])
        
        else:
            # not combining conditioned and unconditioned
            tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
            tokens = torch.Tensor(cond_tokens, dtype=torch.long, device=device)
            context  = clip(tokens)
        
        to_idle(clip) # after you're done with a model you can pass it back to cpu

        if sampler_name == 'ddpm':
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
            print('the time steps should be', n_inference_steps)
            print('they are', sampler.timesteps)
        else:
            raise ValueError('only ddpm sampler allowed for now')
        
        latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)

        if input_image:
            encoder = models['encoder']
            encoder.to(device)

            input_image_tensor = input_image.resize((HEIGHT, WIDTH))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (batch_size, h, w, c) -> (batch_size, c, h, w)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latent_shape, generator=generator, device=device)

            # Run the image through the encoder of the VAE
            latent = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latent = sampler.add_noise(latent, sampler.timesteps[0])

            to_idle(encoder)
        
        else:
            # means we wanna do text to image -> we start from random noise
            latent = torch.randn(latent_shape, generator=generator, device=device)
        
        
        diffusion = models['diffusion']
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            # (batch_size, 4, latent_height, latent_width)
            model_input = latent

            if do_cfg:
                # (batch_size, 4, latent_height, latent_width) -> (2*batch_size, 4, latent_height, latent_width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # predicted amount of noise in the image by the U-Net
            model_output = diffusion(model_input, context, time_embedding)


            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            # once we get how much noise is present in the image the scheduler will remove it
            latent = sampler.step(timestep, latent, model_output) # remove the noise predicted by the U-Net


        to_idle(diffusion)

        decoder = models['decoder']
        decoder.to(device)

        image = decoder(latent)

        to_idle(decoder)

        image = rescale(image, (-1,1), (0,255), clamp=True)
        # (batch_size, channel, h, w) -> (batch_size, h, w, channel)
        image = image.permute(0, 2, 3, 1)
        image = image.to('cpu', torch.uint8).numpy()

        return image


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)



