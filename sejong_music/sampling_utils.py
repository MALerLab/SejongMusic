import torch

def nucleus(probs, p=0.9):
    probs /= (probs.sum() + 1e-5)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    indices = cumulative_probs < p
    if indices.sum() > 0:
        last_index = torch.where(indices[0] == True)[0][-1] + 1
    else:
        last_index = indices.size(0)
    candi_indices = sorted_indices[0,:last_index]
    candi_probs = probs[0][candi_indices]
    candi_probs /= candi_probs.sum()
    word = candi_indices[torch.multinomial(candi_probs, num_samples=1)]
    return word.unsqueeze(0)