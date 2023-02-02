<p align="center">
     <img src="figures/logo.png" alt="logo" width = "600">
     <br/>
</p>


MaskGMT is a text-to-music generation paradigm using a hierarchy bidirectional transformer with tokenize-detokenize neural compression. 
During training, MaskGMT learns to predict randomly masked tokens by attending to tokens in all directions as different residual quantification. 
At inference time, the model begins with generating all main tokens of a music sequence simultaneously, and then refines the residual sequence iteratively conditioned on the previous generation. 

## Usage

### Text-Conditional Generator
A text-to-music model that conditions the generation with `t5-base` text embeddings. 

```py
from maskgmt import MaskGmtTransformer, MaskGmt, TokenCritic

transformer = MaskGmtTransformer(
    num_tokens = 1024,        # must be same as encodec codebook size above
    seq_len = 750,            # must be down-sampling length
    dim = 512,                # model dimension
    n_q = 8,                  # down-sampling quantification dim
    depth = 2,                # depth of transformer block
    dim_head = 64,            # attention head dimension
    heads = 8,                # attention heads,
    ff_mult = 4,              # feedforward expansion factor
)

critic = TokenCritic(
    num_tokens = 1024,
    seq_len = 750,
    dim = 512,
    depth = 6,
)

maskgmt_model = MaskGmt(
    transformer = transformer,
    cond_drop_prob = 0.25,
    token_critic = critic, 
    self_token_critic = False,
)


texts = ['I have a pencil', 'I have a pen'] 
music = torch.ones(2, 8, 750).long() 

# training
loss = maskgmt_model(music_ids=music, texts=texts) 
loss.backward()

# infer 
music_tokens = maskgmt_model.generate(
     texts = texts,
     cond_scale = 3.,
)

```


### Music Tokenizer and DeTokenizer 
Transfer wav formation of music to torch code and revserse back.

```py
from encodec import EncodecModel 
from encodec.utils import convert_audio, save_audio

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0) 

path = 'test.wav' 

# transfer to code
wav, sr = torchaudio.load(path) 
wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0) 
with torch.no_grad():
    encoded_frames = model.encode(wav) 
codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)

# transfer to wav 
with torch.no_grad(): 
    frames = model.decode([(codes, None)]).squeeze(0)
save_audio(frames, path, sample_rate= model.sample_rate)

```


## Cases 
We provide the trained ckpt in the google drive.
Some well-generated music cases can be found in google drive.



## Acknowledge 

Our implementation is based on [encodec](https://github.com/facebookresearch/encodec), [MaskGiT](https://github.com/lucidrains/muse-maskgit-pytorch), and [Huggingface](https://github.com/huggingface/transformers). Thanks for their clear and beautiful code. 
