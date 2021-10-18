import time

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers

from mesh_transformer.checkpoint import read_ckpt_lowmem
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

import argparse

class GPTJModel:
    def __init__(self):
        params = {
            "layers": 28,
            "d_model": 4096,
            "n_heads": 16,
            "n_vocab": 50400,
            "norm": "layernorm",
            "pe": "rotary",
            "pe_rotary_dims": 64,
            "seq": 2048,
            "cores_per_replica": 8,
            "per_replica_batch": 1,
        }

        per_replica_batch = params["per_replica_batch"]
        cores_per_replica = params["cores_per_replica"]
        self.seq = params["seq"]

        params["sampler"] = nucleaus_sample
        # scale the optimizer to 0 from the model (as we don't need them for inference)
        params["optimizer"] = optax.scale(0)

        mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
        devices = np.array(jax.devices()).reshape(mesh_shape)

        maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

        print('Loading checkpoint..')

        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        self.total_batch = per_replica_batch * jax.device_count() // cores_per_replica
        self.network = CausalTransformer(params)
        self.network.state = read_ckpt_lowmem(self.network.state, "./checkpoints/step_383500/", devices.shape[1])
        self.network.state = self.network.move_xmap(self.network.state, np.zeros(cores_per_replica))

    def infer(self, context, top_p=0.9, temp=1.0, gen_len=512):
        tokens = self.tokenizer.encode(context)

        provided_ctx = len(tokens)
        pad_amount = self.seq - provided_ctx

        padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
        batched_tokens = np.array([padded_tokens] * self.total_batch)
        length = np.ones(self.total_batch, dtype=np.uint32) * len(tokens)

        start = time.time()
        output = self.network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(self.total_batch) * top_p, "temp": np.ones(self.total_batch) * temp})

        samples = []
        decoded_tokens = output[1][0]

        for o in decoded_tokens[:, :, 0]:
            samples.append(f"\033[1m{context}\033[0m{self.tokenizer.decode(o)}")

        print(f"completion done in {time.time() - start:06}s")

        return samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT J Options.')

    parser.add_argument('text', action='store', type=str,
                        help='Text GPT J will use as context')
    parser.add_argument('--top_p', action='store',
                        type=float, default=0.9,
                        help='Adjust top_p option - scale 0.1')
    parser.add_argument('--temp', action='store',
                        type=float, default=1.0,
                        help='Adjust temp option - scale 0.1')
    parser.add_argument('--gen_len', action='store',
                        type=int, default=512,
                        help='Adjust generated len')

    args = vars(parser.parse_args())

    gptj = GPTJModel()
    print(gptj.infer(context=args['text'], top_p=args['top_p'], temp=args['temp'], gen_len=args['gen_len'])[0])
