from torch.utils.data import *
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

class FakeDataset(Dataset):
    def __init__(self, vocab_size, num_samples, max_length):
        self.vocab_size = vocab_size
        self.samples = num_samples
        self.max_length = max_length
        
    def __len__(self):
        return self.samples
    
    def __getitem__(self, index):
        # 生成一个1到max_length之间的随机数，表示句子的长度
        sentence_length = np.random.randint(1, self.max_length+1)
        
        # 生成一个句子，每个词是一个0到vocab_size之间（不包括vocab_size本身）的随机整数
        sentence = np.random.randint(self.vocab_size, size=sentence_length)
        
        # 把numpy数组转换成PyTorch张量
        sentence_tensor = torch.from_numpy(sentence)
        
        return sentence_tensor
    
def collate_fn(batch):
    sentences = [item for item in batch]

    # Generate attention masks, before padding
    # Here, mask value is 1 for non-padding tokens, and 0 for padding tokens
    attention_masks = [torch.ones_like(sent) for sent in sentences]

    # Pad sentences and attention masks
    padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
    # Reverse the padding (so that padding is in the front of each sequence)
    max_length = padded_sentences.size(1)
    padded_sentences = torch.flip(padded_sentences, [1])
    padded_attention_masks = torch.flip(padded_attention_masks, [1])

    return padded_sentences, padded_attention_masks

def print_model_mem(model):
    # 储存每种数据类型所占的字节数
    type_sizes = {
        torch.float16: 2,
        torch.float32: 4,
        torch.float64: 8,
        torch.uint8: 1,         
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8
    }
    
    total_size = 0
    type_count = set()
    for param in model.parameters():
        type_count.add(param.dtype)
        assert param.dtype in type_sizes, f'Unhandled data type {str(param.dtype)}'
        total_size += torch.prod(torch.tensor(param.shape)) * type_sizes[param.dtype]
    
    total_size_mb = total_size.item() / 1024 / 1024 
    print(f"{total_size_mb} MB memory, {type_count}")

def print_model_size(model, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_params / 1e6} Million params\n")

