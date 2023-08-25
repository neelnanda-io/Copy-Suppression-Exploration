# %%
import os
os.environ["TRANSFORMERS_CACHE"] = "/workspace/cache/"
# %%
from neel.imports import *
from neel_plotly import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_grad_enabled(False)


model: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab
# %%
logits, cache = model.run_with_cache("All's fair in love and")
utils.test_prompt("All's fair in love and", "war", model)
# %%
resid_stack, resid_labels = cache.accumulated_resid(incl_mid=True, apply_ln=True, pos_slice=-1, return_labels=True)
unembed_dirs = model.W_U[:, [model.to_single_token(" love"), model.to_single_token(" war")]]
line((resid_stack @ unembed_dirs).squeeze().T, x=resid_labels, line_labels=["love", "war"], title="Logit Lens Across Layers")
# %%
resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=True, apply_ln=True, pos_slice=-1, return_labels=True)
unembed_dirs = model.W_U[:, [model.to_single_token(" love"), model.to_single_token(" war")]]
line((resid_stack @ unembed_dirs).squeeze().T, x=resid_labels, line_labels=["love", "war"], title="Component DLA for All's fair in love and")

# %%
x = (torch.stack([resid_stack[i] for i in range(len(resid_stack)) if "L10" in resid_labels[i]]) @ unembed_dirs).squeeze().T
line(x, title="Per head DLA in layer 10 on All's fair in love and", line_labels=["love", "war"])
print(x.sum(1))
# %%
nutils.create_vocab_df(model.W_out[11, 1431] @ model.W_U)['logit'].mean()
# %%
data = load_dataset("stas/openwebtext-10k")
dataset = utils.tokenize_and_concatenate(data["train"], model.tokenizer, max_length=256)
all_tokens = dataset["tokens"]
# %%
tokens = all_tokens[::17][:32].cuda()
# %%
logits, cache = model.run_with_cache(tokens)
# %%
pattern = cache["pattern", 10][:, 7, :, :]
max_non_bos_pattern, argmax_non_bos_pattern = pattern[:, :, 1:].max(-1)
argmax_non_bos_pattern+=1
argmax_token = tokens[np.arange(tokens.shape[0])[:, None], argmax_non_bos_pattern]

head_dla = cache["z", 10][:, :, 7, :] @ model.W_O[10, 7] @ model.W_U / cache["scale"]

final_logits = logits
final_log_probs = logits.log_softmax(dim=-1)

mid_dla = cache["resid_post", 9]
mid_logits = (mid_dla / cache["scale"]) @ model.W_U
mid_log_probs = mid_logits.log_softmax(dim=-1)
# %%
print("Final loss", final_log_probs[:, :-1, :].gather(-1, tokens[:, 1:, None]).mean())
mean_ablate_head_dla = head_dla.mean([0, 1], keepdim=True)
print("Final loss zero ablate head", (final_logits - head_dla).log_softmax(-1)[:, :-1, :].gather(-1, tokens[:, 1:, None]).mean())
print("Final loss mean ablate head", (final_logits - head_dla + mean_ablate_head_dla).log_softmax(-1)[:, :-1, :].gather(-1, tokens[:, 1:, None]).mean())
print("Mid loss", mid_log_probs[:, :-1, :].gather(-1, tokens[:, 1:, None]).mean())
print("Mid loss plus head", (mid_logits + head_dla).log_softmax(dim=-1)[:, :-1, :].gather(-1, tokens[:, 1:, None]).mean())
print("Mid loss with mean ablation", (mid_logits + (final_logits - mid_logits).mean([0, 1])).log_softmax(dim=-1)[:, :-1, :].gather(-1, tokens[:, 1:, None]).mean())
print("Mid loss plus head with mean ablation", (mid_logits + (head_dla - mean_ablate_head_dla) + (final_logits - mid_logits).mean([0, 1])).log_softmax(dim=-1)[:, :-1, :].gather(-1, tokens[:, 1:, None]).mean())
print("Mid loss plus head with mean ablation (except 10.7)", (mid_logits + (head_dla) + (final_logits - mid_logits).mean([0, 1])).log_softmax(dim=-1)[:, :-1, :].gather(-1, tokens[:, 1:, None]).mean())
print("Mid loss no LN Freeze", (model.ln_final(mid_dla) @ model.W_U).log_softmax(dim=-1)[:, :-1, :].gather(-1, tokens[:, 1:, None]).mean())
# %%
# %%
scatter(x=mid_log_probs[:, :-1, :].gather(-1, tokens[:, 1:, None]).flatten(), y=final_log_probs[:, :-1, :].gather(-1, tokens[:, 1:, None]).flatten())
# %%
histogram(max_non_bos_pattern.flatten())
# %%
argmax_token_mid_logits = mid_logits.argmax(dim=-1)
(argmax_token_mid_logits == argmax_token).float().mean()
# %%
rank_of_attended_token = (mid_logits >= mid_logits.gather(-1, argmax_token[:, :, None])).sum(-1).flatten()
# %%
argmax_token_str = nutils.list_flatten([model.to_str_tokens(t) for t in argmax_token])
# %%
token_df = nutils.make_token_df(tokens)
token_df["rank_of_attended_token"] = to_numpy(rank_of_attended_token).flatten()
token_df["max_non_bos_pattern"] = to_numpy(max_non_bos_pattern).flatten()
token_df["argmax_non_bos_pattern"] = to_numpy(argmax_non_bos_pattern).flatten()
token_df["argmax_token_pattern"] = to_numpy(argmax_token).flatten()
token_df["argmax_token_pattern_str"] = to_numpy(argmax_token_str).flatten()

# %%
token_df["argmax_token_mid_logit"] = to_numpy(mid_logits.gather(-1, argmax_token[:, :, None]).flatten())
token_df["argmax_token_mid_log_prob"] = to_numpy(mid_log_probs.gather(-1, argmax_token[:, :, None]).flatten())
token_df["argmax_token_final_logit"] = to_numpy(final_logits.gather(-1, argmax_token[:, :, None]).flatten())
token_df["argmax_token_final_log_prob"] = to_numpy(final_log_probs.gather(-1, argmax_token[:, :, None]).flatten())
token_df["argmax_token_head_dla"] = to_numpy(head_dla.gather(-1, argmax_token[:, :, None]).flatten())
# %%
token_df.query("rank_of_attended_token == 1").sort_values("argmax_token_mid_log_prob", ascending=False)
# %%
px.scatter(token_df.query("rank_of_attended_token == 1"), x="argmax_token_mid_logit", y="argmax_token_head_dla", hover_name="context").show()
px.scatter(token_df.query("rank_of_attended_token == 1"), x="argmax_token_mid_log_prob", y="argmax_token_head_dla", hover_name="context").show()
# %%
px.scatter(token_df.query("rank_of_attended_token == 1"), x="argmax_token_mid_logit", y="max_non_bos_pattern", color="argmax_token_final_log_prob", hover_name="context").show()
px.scatter(token_df.query("rank_of_attended_token == 1"), x="argmax_token_mid_log_prob", y="max_non_bos_pattern", color="argmax_token_final_log_prob", hover_name="context").show()
# %%
