# TODO

read through "Attention is all you need" paper. Figure out details about multihead attention module. 
- Finish `<YOUR CODE GOES HERE>` parts in `model.py`; *Read comments. They are helpful.*
- Use `einsum` and `einops` for this implementation
- `Transformer-NMT-en-es.ipynb` notebook provides a workflow of NMT.
- `train_tokenizer_en/es.py` scripts are for training Unigram Tokenizer from scratch
- `train.py` script is for training the vanilla seq2seq transformer model.
- **Note:** the `translate` function suffers performance issue / bug: try to identify and fix it.
- Write down your learning and thoughts about Attention and Jax in a separate Markdown file.

## What/How to submit 
submit your work via github:
- fork **a "private" repo, named as "jax-transformer-CNetID"**, from this repo: [How to do that?](https://stackoverflow.com/questions/10065526/github-how-to-make-a-fork-of-public-repository-private)
- include **a seperate Markdown file**, document you thoughts, questions about attention and jax
- add the trained **tokenizer.json** in `vanilla-NMT` folder *(train tokenizer and you will see it)*
- include **ONE `ckpt/state-{timestamp}.pickle`** file *(train model and you will see it)*
- go to repo settings, and **add collaborators: Oaklight**

## When to submit
Deadline is **Oct 10th, noon** for this assignment. **Timestamped by the collaborator invite email.** I will respond to your invites ASAP. 

# Resources about transformer in jax:
- Attention is all you need: https://arxiv.org/abs/1706.03762
- Annotated transformer (Pytorch): https://nlp.seas.harvard.edu/2018/04/03/attention.html
- einsum:
    - videos:
        - Youtube: https://youtu.be/ULY6pncbRY8
        - Bilibili: https://www.bilibili.com/video/BV1ee411g7Sv
    - code snippets: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
    - einops: https://github.com/arogozhnikov/einops/
    - einsum: https://theaisummer.com/einsum-attention/
- Jax:
    - Jax: https://jax.readthedocs.io/en/latest/index.html
        - jax is "functional" https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html#differences-from-numpy
    - Haiku: https://dm-haiku.readthedocs.io/en/latest/index.html
        - Haiku101: Haiku库的基本使用逻辑 - 谷雨的文章 - 知乎 https://zhuanlan.zhihu.com/p/471892075
- reference implementations:
    - haiku:
        this is cleaner but more functional & the transformer is not complete
        - https://github.com/deepmind/dm-haiku/blob/c18be3df5e85796492f2915af261b5517f12bacc/examples/transformer/model.py
        - https://github.com/deepmind/dm-haiku/blob/c18be3df5e85796492f2915af261b5517f12bacc/haiku/_src/attention.py        
    - flax:
        this is more complex and easier to get lost
        - https://github.com/google/flax/blob/6dba29098fba23a457e87f104bfef2704dbf54cd/examples/wmt/models.py
        - https://github.com/google/flax/blob/cc88a73f5cf3d5970981c104364bc5864841db1a/flax/linen/attention.py
    - elegy:
        perhaps use it for training loop
        - https://github.com/poets-ai/elegy
    - https://nn.labml.ai/transformers/mha.html
