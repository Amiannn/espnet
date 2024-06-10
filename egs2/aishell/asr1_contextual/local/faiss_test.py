import torch
import faiss
import numpy as np


faiss_gpu_id = torch.cuda.current_device()
faiss_res    = faiss.StandardGpuResources()

print(f'faiss_gpu_id: {faiss_gpu_id}')
print(f'faiss_res: {faiss_res}')

B, D = 5, 10
datas = np.random.rand(B, D)

index = faiss.IndexFlatIP(D)
index = faiss.index_cpu_to_gpu(faiss_res, faiss_gpu_id, index)
index.add(datas)

query = np.random.rand(1, D)
D, I  = index.search(query, 2)
print(I)