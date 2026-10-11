import torch
import glob
files = glob.glob('/mnt/pune/teacher_cache/*.pt')
if files:
    data = torch.load(files[0], weights_only=True)
    print(data.keys())
else:
    print('No files')
