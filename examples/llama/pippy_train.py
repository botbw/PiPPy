import torch
from model import Transformer, ModelArgs
from torch.utils.data import *
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm
from utils import *
import os
import random
import logging

from pippy import *
from pippy.microbatch import TensorChunkSpec, LossReducer

class OutputLossWrapper(LossWrapper):
    def __init__(self, module, loss_fn=None):
        super().__init__(module, loss_fn)

    def forward(self, input, start_pos):
        out = self.module(input, start_pos)
        out_ = torch.ones_like(out)
        return out_

def run_main(rank, args):
    logging.info(f"Running main on rank {rank}")
    
    device = torch.device("cuda" if args.cuda else "cpu")
    model = Transformer(ModelArgs).to(device)
    print_model_size(model)
    print_model_mem(model)

    split_policy = split_into_equal_size(args.world_size - 1)
    pipe = Pipe.from_tracing(model, split_policy=split_policy)

    all_worker_ranks = list(range(1, args.world_size))  # exclude master rank = 0
    chunks = len(all_worker_ranks)
    
    output_chunk_spec = (TensorChunkSpec(0), LossReducer(torch.tensor(0.0), lambda a, b: a + b))

    pipe_driver = PipelineDriverFillDrain(
        pipe, chunks,
        len(all_worker_ranks),
        all_ranks=all_worker_ranks,
        output_chunk_spec=output_chunk_spec
    )
    # optimizer = pipe_driver.instantiate_optimizer(torch.optim.Adam, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    optimizer = pipe_driver.instantiate_optimizer(torch.optim.SGD, lr=1e-3)
    
    dataset = FakeDataset(vocab_size=ModelArgs.vocab_size, num_samples=args.dataset_size, max_length=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    pipe_driver.train()
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
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 5)))
    parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', str(random.randint(29500, 29600))))
    parser.add_argument('--max_seq_len', default=256, type=int)
    parser.add_argument('--dataset_size', default=1000, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--cuda', default=int(torch.cuda.is_available()), type=int)

    run_pippy(run_main, parser.parse_args())