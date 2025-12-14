import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()  
    torch.cuda.synchronize()