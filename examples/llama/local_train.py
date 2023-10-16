from model import Transformer, ModelArgs
from torch.utils.data import *
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm
from utils import *

def main(args):
    device = torch.device(args.device)
    model = Transformer(ModelArgs).to(device)
    print_model_size(model)
    print_model_mem(model)
    
    dataset = FakeDataset(vocab_size=ModelArgs.vocab_size, num_samples=args.dataset_size, max_length=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    
    model.train()
    for epoch in range(args.epochs):
        for step, (seq, _) in enumerate(tqdm(dataloader, dynamic_ncols=True)):
            seq = seq.to(device)
            optimizer.zero_grad()
            start_pos = 0
            h, freqs_cis, mask = model.pre_forward(seq, start_pos)
            out = model(h, start_pos, freqs_cis, mask)
            out_grad = torch.ones_like(out)
            out.backward(out_grad)
            optimizer.step()
            tqdm.write(f"Epoch {epoch} Step {step} Loss {out.detach().sum().item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--dataset_size', default=1000, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    main(parser.parse_args())
