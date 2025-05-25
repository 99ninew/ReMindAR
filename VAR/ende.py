import torch
import torchvision.transforms as T
from PIL import Image
import random
import numpy as np
import os
import requests
from typing import List, Optional, Tuple, Union
# from models import VQVAE, build_vae_var
from VAR.models import VQVAE, build_vae_var
from torch.nn.parallel import DistributedDataParallel as DDP
from VAR.utils.data import pil_loader, normalize_01_into_pm1
# from utils.data import pil_loader, normalize_01_into_pm1
import torchvision
from PIL import Image as PImage
from torchvision.transforms import transforms
# from VAR.models.var import AdaLNSelfAttn, sample_with_top_k_top_p_, gumbel_softmax_with_rng


class VARImagePipeline:
    def __init__(
        self,
        model_depth: int = 36,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hf_home: str = "https://huggingface.co/FoundationVision/var/resolve/main",
        model_dir: str = "/data/chenxiao/VAR/models",
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    ):
        self.device = device
        if model_depth != 36:
            patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        self.patch_nums = patch_nums
        self.model_depth = model_depth
        self._setup_models(hf_home, model_dir)
        # self._setup_transforms()

    def _setup_models(self, hf_home: str, model_dir: str):
        
        os.makedirs(model_dir, exist_ok=True)
        
        vae_ckpt = os.path.join(model_dir, "vae_ch160v4096z32.pth")
        var_ckpt = os.path.join(model_dir, f"var_d{self.model_depth}.pth")
        
        if not os.path.exists(vae_ckpt):
            self._download_file(f"{hf_home}/vae_ch160v4096z32.pth", vae_ckpt)
        if not os.path.exists(var_ckpt):
            self._download_file(f"{hf_home}/var_d{self.model_depth}.pth", var_ckpt)

        FOR_512_px = self.model_depth == 36
        print(FOR_512_px)
        self.vae, self.var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,
            device=self.device, patch_nums=self.patch_nums,
            num_classes=1000, depth=self.model_depth, shared_aln=FOR_512_px,
        )

        self.vae.load_state_dict(torch.load(vae_ckpt, map_location="cpu"), strict=True)
        self.var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=True)
        
        self.vae.eval()
        self.var.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.var.parameters():
            p.requires_grad_(False)

    def _download_file(self, url: str, local_path: str):
        print(f"Downloading {url} to {local_path}")
        response = requests.get(url, stream=True)
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


    def encode(self, images: Union[Image.Image, List[Image.Image], torch.Tensor], st: int = None) -> List[torch.Tensor]:

        # print(images)
        B = len(images) if isinstance(images, list) else images.shape[0] if isinstance(images, torch.Tensor) else 1
        # print(B)
        input_img = torch.stack([normalize_01_into_pm1(transforms.ToTensor()(img) if not isinstance(img, torch.Tensor) else img).to(device=self.device) for img in images], dim=0)
        # print(input_img.shape)
        # input_img = torch.stack([normalize_01_into_pm1(transforms.ToTensor()(img)).to(device=self.device) for img in images], dim=0)
        input_img_tokens = self.vae.img_to_idxBl(input_img, self.var.patch_nums)
        res = []
        for si in range(st + 1):
            pn = self.var.patch_nums[si] 
            # lookup
            BChw = self.var.vae_quant_proxy[0].embedding(input_img_tokens[si]).transpose_(1, 2).reshape(B, self.var.Cvae, pn, pn)
            res.append(BChw)
        return res


    
    def decode(
        self,
        latent: List[torch.Tensor],
        label: Optional[Union[int, torch.Tensor]] = None,
        cfg: float = 3.0,
        top_k: int = 900,
        top_p: float = 0.95,
        seed: Optional[int] = 42,
        st: int = None,
    ) -> List[Image.Image]:
        # class_labels = (980, 980, 437, 437, 22, 22, 562, 562)
        # class_labels = (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000)
        # B = len(class_labels)
        # label_B: torch.LongTensor = torch.tensor(class_labels, device=self.device)
        
        
        # if label is None:
        #     label_B = None
        # elif isinstance(label, int):
        #     label_B = torch.full((B,), label, device=self.device)
        # else:
        #     label_B = label.to(self.device)
        
        # label_B = 980
        B, C, H, W = latent[0].shape
        # seed = 42
        if seed is not None:
            torch.manual_seed(seed)
        
        # label = 980
        if label is None:
            class_label = 1000
            # label = torch.tensor([class_label], device=self.device)
            label = torch.full((B,), class_label, device=self.device, dtype=torch.long)
        elif isinstance(label, int):
            label = torch.full((B,), label, device=self.device, dtype=torch.long)
        else:
            label = label.to(self.device)
            if label.ndim == 0:
                label = label.expand(B)
            elif label.shape[0] != B:
                label = label.expand(B)
        
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

        for i in range(len(latent)):
            ph, pw = self.var.patch_nums[i], self.var.patch_nums[i]
            z_NC = latent[i].permute(0, 2, 3, 1).reshape(-1, C)
            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.var.vae_quant_proxy[0].embedding.weight.data.square(), dim=1, keepdim=False)
            d_no_grad.addmm_(z_NC, self.var.vae_quant_proxy[0].embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)
            idx_Bhw = idx_N.view(B, ph, pw)
            latent[i] = self.var.vae_quant_proxy[0].embedding(idx_Bhw).permute(0, 3, 1, 2)
        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                recon_B3HW = self.var.autoregressive_infer_cfg_with_mask(
                    # self.var,
                    B=B, label_B=label, cfg=cfg, top_k=top_k, top_p=top_p, g_seed=0, more_smooth=True,
                    latent=latent, st=st
                )

        chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).clone().mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        # print(chw.dtype)
        # print(chw[0].shape)
        # print(chw)
        # print(chw.shape)
        # print(chw.type)
        return chw
        # return [self.postprocess(img) for img in recon_B3HW]

    def reconstruct(
        self,
        images: Union[Image.Image, List[Image.Image]],
        label: Optional[Union[int, torch.Tensor]] = None,
        **decode_kwargs
    ) -> List[Image.Image]:
        
        latent = self.encode(images, st=5)
        return self.decode(latent, st=5)

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        label: Optional[Union[int, torch.Tensor]] = None,
        **kwargs
    ) -> List[Image.Image]:
        
        return self.reconstruct(images, label=label, **kwargs)



if __name__ == "__main__":
    
    pipeline = VARImagePipeline(model_depth=36)
    
    
    # img = Image.open("image1.jpg").convert("RGB")
    
    # width, height = img.size
    # left = (width - 256) / 2
    # top = (height - 256) / 2
    # right = (width + 256) / 2
    # bottom = (height + 256) / 2

    
    # img = img.crop((left, top, right, bottom))
    
    
    # reconstructed_imgs = pipeline(img)
    # images_to_be_edited = ["image1.jpg", "image2.jpg", "image3.jpg"]
    images_to_be_edited = ["image1.jpg"]
    # img_to_be_edited = "image2.jpg"
    # img = pil_loader(img_to_be_edited)
    img = []
    for img_path in images_to_be_edited:
        im = pil_loader(img_path)
        im = im.resize((512, 512), PImage.LANCZOS)
        img.append(im)
    # img = img.resize((512, 512), PImage.LANCZOS)
    # img = [img]
    reconstructed_imgs = pipeline(img)
    reconstructed_imgs.save("outputs/result.png")
    # reconstructed_imgs = [img.resize((256, 256), PImage.LANCZOS) for img in reconstructed_imgs]
    
    # for i, img in enumerate(reconstructed_imgs):
    #     image = Image.fromarray(img)
    #     image.save(f"outputs/result_{i}.png")
    # for i, img in enumerate(reconstructed_imgs):
    #     image = Image.fromarray(img)
    #     image.save(f"outputs/result_{i}.png")       
    
    # reconstructed_imgs[0].save("result.png")

    # reconstructed_imgs[0].show()