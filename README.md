# gpt-j-docker
simple implementation to run gpt-j anywhere

# Running
inside terminal execute <br />
docker build -t gpt-j . <br />
docker run -i -t gpt-j <br />

# Dependencies
Note this library has some specific requirements for JAX version. Specifically, to use the v1 models (including GPT-J 6B), jax==0. <br />
2.12 is required. This in turn depends on jaxlib==0.1.68. If this is not done, you will get cryptic xmap errors <br />
refer to https://github.com/kingoflolz/mesh-transformer-jax <br />
