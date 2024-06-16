import torch
from torch import nn, Tensor
from transformers import AutoModelForCausalLM, AutoConfig, AutoModel
from MeshAnything.miche.encode import load_model
from MeshAnything.models.shape_opt import ShapeOPTConfig
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack
import torch.nn.functional as F

class NoiseResistantDecoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pad_id = -1
        self.num_quantizers = 3

        self.discrete_num = 128
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim

        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.num_hidden_layers = 6
        self.decoder= AutoModel.from_config(config=config).to_bettertransformer().encoder
        self.n_embd = self.decoder.config.hidden_size

        self.pos_embedding = nn.Embedding(18000, self.n_embd)
        self.layernorm = nn.LayerNorm(self.n_embd)
        self.point_layernorm = nn.LayerNorm(self.n_embd)

        self.cond_length = 257
        self.cond_dim = 768
        self.point_pe = nn.Embedding(self.cond_length, self.n_embd)
        self.cond_proj = nn.Linear(self.cond_dim, self.n_embd)
        self.cond_head_proj = nn.Linear(self.cond_dim, self.n_embd)

        self.project_down_codebook = nn.Linear(self.codebook_dim * 3, self.n_embd)
        self.to_coor_logits = nn.Sequential(
            nn.Linear(self.n_embd, self.discrete_num * 9),
            Rearrange('... (v c) -> ... v c', v = 9)
        )
    def process_point_feature(self, encode_feature):
        point_feature = torch.zeros(encode_feature.shape[0], self.cond_length, self.n_embd, device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        point_feature[:, 0] = self.cond_head_proj(encode_feature[:, 0])
        point_feature[:, 1:] = self.cond_proj(encode_feature[:, 1:])

        point_feature = self.point_layernorm(point_feature + self.point_pe.weight[None, :point_feature.shape[1]])
        return point_feature

    def forward(self, input_ids, input_embeds, point_feature = None):
        input_ids = input_ids.reshape(input_ids.shape[0], -1)
        point_feature = self.process_point_feature(point_feature)

        face_embeds = rearrange(input_embeds, 'b (nf nv) d -> b nf (nv d)', nv = 3)
        face_embeds = self.project_down_codebook(face_embeds)

        face_mask = reduce(input_ids != self.pad_id, 'b (nf nv q) -> b nf', 'all', nv = 3, q = self.num_quantizers)
        face_embeds[~face_mask] = 0

        face_embeds = self.layernorm(face_embeds + self.pos_embedding.weight[None, :face_embeds.shape[1]])

        outputs = self.decoder(
            hidden_states=torch.concatenate([point_feature, face_embeds], dim=1),
        )
        decoded = outputs.last_hidden_state[:, self.cond_length:]       # batch x nfaces x dim
        decoded = decoded.masked_fill(~face_mask.unsqueeze(-1), 0.)

        # batch x nfaces x 9 -> batch x nfaces x 3 x 3
        pred_face_logits = self.to_coor_logits(decoded)     # batch x nfaces x 9 x ndiscrete
        pred_face_coords = rearrange(pred_face_logits.argmax(dim = -1), '... (v c) -> ... v c', v = 3)

        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.discrete_num,
            low = -0.5,
            high = 0.5
        )
        continuous_coors = continuous_coors.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1 1'), float('nan'))

        return continuous_coors

class MeshAnything(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.point_encoder = load_model(ckpt_path=None)
        self.tokenizer = NoiseResistantDecoder(args)

        self.num_quantizers = 3
        self.face_per_token = self.num_quantizers * 3
        self.cond_length = 257
        self.cond_dim = 768
        self.max_length = args.n_max_triangles * self.face_per_token + 2 + self.cond_length

        self.config = ShapeOPTConfig.from_pretrained(
            args.llm,
            n_positions=18259,
            max_position_embeddings=18259,
            vocab_size=self.tokenizer.codebook_size + 3,
            _attn_implementation="flash_attention_2"
        )
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2
        self.config.bos_token_id = self.bos_token_id
        self.config.eos_token_id = self.eos_token_id
        self.config.pad_token_id = self.pad_token_id
        self.config.quantize_codebook_dim = self.tokenizer.codebook_dim
        self.config.face_per_token = self.face_per_token
        self.config._attn_implementation="flash_attention_2"
        self.config.cond_length = self.cond_length
        if self.config.word_embed_proj_dim != self.config.hidden_size:
            self.config.word_embed_proj_dim = self.config.hidden_size
        self.transformer = AutoModelForCausalLM.from_config(
            config=self.config, use_flash_attention_2 = True
        )
        self.transformer.to_bettertransformer()
        self.transformer.model.decoder.quantize_codebooks = nn.Parameter(torch.zeros(1, self.tokenizer.codebook_size, self.tokenizer.codebook_dim))

        self.cond_head_proj = nn.Linear(self.cond_dim, self.config.word_embed_proj_dim)
        self.cond_proj = nn.Linear(self.cond_dim * 2, self.config.word_embed_proj_dim)

        self.eval()

    def process_point_feature(self, point_feature):
        encode_feature = torch.zeros(point_feature.shape[0], self.cond_length, self.config.word_embed_proj_dim,
                                    device=self.cond_head_proj.weight.device, dtype=self.cond_head_proj.weight.dtype)
        encode_feature[:, 0] = self.cond_head_proj(point_feature[:, 0])
        shape_latents = self.point_encoder.to_shape_latents(point_feature[:, 1:])
        encode_feature[:, 1:] = self.cond_proj(torch.cat([point_feature[:, 1:], shape_latents], dim=-1))

        return encode_feature

    @torch.no_grad()
    def forward(self, pc_normal, sampling=False) -> dict:
        batch_size = pc_normal.shape[0]
        point_feature = self.point_encoder.encode_latents(pc_normal)
        processed_point_feature = self.process_point_feature(point_feature)

        generate_length = self.max_length - self.cond_length
        net_device = next(self.parameters()).device
        outputs = torch.ones(batch_size, generate_length).long().to(net_device) * self.eos_token_id
        if not sampling:
            results = self.transformer.generate(
                inputs_embeds=processed_point_feature,
                max_new_tokens=generate_length,  # all faces plus two
                num_beams=1,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
            )
        else:
            results = self.transformer.generate(
                inputs_embeds = processed_point_feature,
                max_new_tokens=generate_length, # all faces plus two
                do_sample=True,
                top_k=50,
                top_p=0.95,
                bos_token_id = self.bos_token_id,
                eos_token_id = self.eos_token_id,
                pad_token_id = self.pad_token_id,
            )
        assert results.shape[1] <= generate_length # B x ID  bos is not included since it's predicted
        outputs[:, :results.shape[1]] = results
        # batch x ntokens ====> batch x ntokens x D
        outputs = outputs[:, 1: -1]

        outputs[outputs == self.bos_token_id] = self.tokenizer.pad_id
        outputs[outputs == self.eos_token_id] = self.tokenizer.pad_id
        outputs[outputs == self.pad_token_id] = self.tokenizer.pad_id

        outputs[outputs != self.tokenizer.pad_id] -= 3
        code_embed = self.get_codes(outputs)
        decoder_output = self.tokenizer(outputs, code_embed, point_feature=point_feature)

        return decoder_output

    def get_codes(self, indices):
        indices = indices.reshape(indices.shape[0], -1)

        indices = rearrange(indices, 'b (n q) -> b n q', q=self.num_quantizers)

        batch, quantize_dim = indices.shape[0], indices.shape[-1]
        # may also receive indices in the shape of 'b h w q' (accept_image_fmap)

        indices, ps = pack([indices], 'b * q')

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # take care of quantizer dropout

        mask = indices == -1.
        indices = indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        # dummy implementation for shared codebook
        all_codes = self.transformer.model.decoder.quantize_codebooks[0][indices]
        all_codes = all_codes.permute(2, 0, 1, 3)

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(rearrange(mask, 'b n q -> q b n 1'), 0.)

        # if (accept_image_fmap = True) then return shape (quantize, batch, height, width, dimension)

        codes, = unpack(all_codes, ps, 'q b * d')

        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return codes_summed

def undiscretize(
    t,
    low,
    high,
    num_discrete
) -> Tensor:
    t = t.float()

    t /= num_discrete
    return t * (high - low) + low
