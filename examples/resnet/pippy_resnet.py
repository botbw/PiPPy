# Copyright (c) Meta Platforms, Inc. and affiliates
import argparse
import os
import logging
from functools import reduce

import torch
from torch import optim
from torch.nn.functional import cross_entropy
from torchvision.models import resnet18
from tqdm import tqdm  # type: ignore

import torch.fx
from pippy import run_pippy, split_into_equal_size
from pippy.IR import MultiUseParameterConfig, Pipe, LossWrapper, PipeSplitWrapper, annotate_split_points
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.events import EventsContext
from pippy.microbatch import sum_reducer, TensorChunkSpec
from pippy.visualizer import events_to_json
from pippy.logging import setup_logger
from resnet import ResNet18
import random

PROFILING_ENABLED = True
CHECK_NUMERIC_EQUIVALENCE = True

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

torch.fx.Tracer.proxy_buffer_attributes = True

def run_master(_, args):
    print("Using schedule:", args.schedule)
    print("Using device:", args.device)

    number_of_workers = 2
    all_worker_ranks = list(range(1, 1 + number_of_workers))  # exclude master rank = 0
    chunks = 1
    batch_size = args.batch_size * chunks

    model = resnet18()

    class OutputLossWrapper(LossWrapper):
        def __init__(self, module):
            super().__init__(module, None)

        def forward(self, input):
            output = self.module(input)
            loss = output.mean()
            # Here we use a dict with the "loss" keyword so that PiPPy can automatically find the loss field when
            # generating the backward pass
            return {"output": output, "loss": loss}

    wrapper = OutputLossWrapper(model)

    split_policy = split_into_equal_size(2)
    pipe = Pipe.from_tracing(wrapper, split_policy=split_policy)
    pipe.to(args.device)

    output_chunk_spec = (TensorChunkSpec(0), sum_reducer)
    pipe_driver: PipelineDriverBase = schedules[args.schedule](pipe, chunks,
                                                               len(all_worker_ranks),
                                                               all_ranks=all_worker_ranks,
                                                               output_chunk_spec=output_chunk_spec,
                                                               _record_mem_dumps=bool(args.record_mem_dumps),
                                                               checkpoint=bool(args.checkpoint))

    optimizer = pipe_driver.instantiate_optimizer(optim.Adam, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    random_input = torch.rand(args.batch_size, 3, 224, 224).to(args.device)

    optimizer.zero_grad(True)
    outp, _ = pipe_driver(random_input)
    print(outp.mean())

    optimizer.step()

    outp, _ = pipe_driver(random_input)
    print(outp.mean())
    print('Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 3)))
    parser.add_argument('--local-rank', type=int, required=True)
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', 'localhost'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', str(random.randint(29500, 29600))))

    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('-s', '--schedule', type=str, default=list(schedules.keys())[0], choices=schedules.keys())
    parser.add_argument('--replicate', type=int, default=int(os.getenv("REPLICATE", '0')))
    parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
    parser.add_argument('--visualize', type=int, default=0, choices=[0, 1])
    parser.add_argument('--record_mem_dumps', type=int, default=0, choices=[0, 1])
    parser.add_argument('--checkpoint', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    args.rank = args.local_rank

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    setup_logger(logging.DEBUG)
    run_pippy(run_master, args)
