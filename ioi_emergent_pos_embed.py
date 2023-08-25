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
prompts = [
    "When John and Mary went to the store, John gave a bottle of milk to",
    "When Mary and John went to the store, John gave a bottle of milk to",
    "When John and Mary went to the store, Mary gave a bottle of milk to",
    "When Mary and John went to the store, Mary gave a bottle of milk to",
]
answers = [
    (" Mary", " John"),
    (" Mary", " John"),
    (" John", " Mary"),
    (" John", " Mary"),
]
answer_tokens = torch.tensor([[model.to_single_token(s) for s in t] for t in answers]).cuda()
prompt_tokens = model.to_tokens(prompts).cuda()

JOHN = model.to_single_token(" John")
MARY = model.to_single_token(" Mary")
# %%
logits, cache = model.run_with_cache(prompt_tokens)
logits[:, -1, [JOHN, MARY]]
# %%
bos_key = cache["key", 9][:, 0, 9, :]
first_name_key = cache["key", 9][:, 2, 9, :] - bos_key
second_name_key = cache["key", 9][:, 4, 9, :] - bos_key
diff_key = second_name_key - first_name_key
final_query = cache["query", 9][:, -1, 9, :]
LABELS = ["ABA", "BAA", "ABB", "BAB"]
imshow([bos_key @ final_query.T, first_name_key @ final_query.T, second_name_key @ final_query.T, diff_key @ final_query.T], facet_col=0, facet_labels=["BOS", "First", "Second", "Diff"], x=LABELS, y=LABELS, xaxis="Dest", yaxis="Src")
# %%
is_first_query = (final_query[1] + final_query[2] - final_query[0] - final_query[3])
is_first_query = is_first_query / is_first_query.norm()
is_first_key = (first_name_key - second_name_key).mean(0)
is_first_key = is_first_key / is_first_key.norm()
is_first_key @ is_first_query
# %%
(diff_key * final_query).sum(-1) / final_query.norm(dim=-1) / diff_key.norm(dim=-1)
# %%
for expand_neurons in [True, False]:
    resid_stack, resid_labels = cache.get_full_resid_decomposition(9, expand_neurons=expand_neurons, apply_ln=True, pos_slice=[2, 4], return_labels=True)
    resid_norms = (resid_stack[:, :, 0, :].norm(dim=-1).mean(-1) + resid_stack[:, :, 1, :].norm(dim=-1).mean(-1))/2
    diff_stack = resid_stack[:, :, 0, :] - resid_stack[:, :, 1, :]
    diff_stack = diff_stack.mean(1)
    diff_stack.shape
    line(diff_stack.norm(dim=-1) / resid_norms, x=resid_labels, title="Frac of norm from difference")

    line(diff_stack @ model.W_K[9, 9] @ is_first_query, x=resid_labels, title="Query Projection of Difference")
    
# %%
resid_stack, resid_labels = cache.get_full_resid_decomposition(3, mlp_input=True, expand_neurons=True, apply_ln=True, pos_slice=[2, 4], return_labels=True)

diff_stack = resid_stack[:, :, 0, :] - resid_stack[:, :, 1, :]
diff_stack = diff_stack.mean(1)


line(diff_stack @ model.W_in[3, :, 563], x=resid_labels, title="Query Projection of Difference")

# %%
ni = 563
ni2 = 2610
win1 = model.W_in[3, :, ni]
win2 = model.W_in[3, :, ni2]
wout1 = model.W_out[3, ni, :]
wout2 = model.W_out[3, ni2, :]
print(model.b_in[3, ni], model.b_in[3, ni2])
print(win1.norm(), win2.norm(), win1 @ win2 / win1.norm() / win2.norm())
print(wout1.norm(), wout2.norm(), wout1 @ wout2 / wout1.norm() / wout2.norm())
# %%
tokens = model.to_tokens(model.sample_datapoint())
logits, cache = model.run_with_cache(tokens)
scatter(x=cache["post", 3][0, :, ni], y=cache["post", 3][0, :, ni2], hover=nutils.make_token_df(tokens)["context"])
# %%
model = HookedTransformer.from_pretrained("gpt2-medium")
utils.test_prompt("I went shopping to buy milk, eggs and", "eggs", model)
utils.test_prompt("I went shopping to buy milk, cheese and", "eggs", model)
utils.test_prompt("I went shopping to buy milk, bread and", "eggs", model)
utils.test_prompt("I went shopping to buy milk, butter and", "eggs", model)
utils.test_prompt("I went shopping to buy milk, eggs,", "eggs", model)
utils.test_prompt("I went shopping to buy milk, cheese,", "eggs", model)
utils.test_prompt("I went shopping to buy milk, bread,", "eggs", model)
utils.test_prompt("I went shopping to buy milk, butter,", "eggs", model)
# %%
logits, cache = model.run_with_cache("I went shopping to buy milk, eggs and")
unembed_dir = model.W_U[:, model.to_single_token(" eggs")]
resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
eggs_dla = resid_stack @ unembed_dir
line((resid_stack @ unembed_dir).squeeze().T, x=resid_labels, title="DLA of eggs Across Layers")
# %%
eigenvals = model.OV.eigenvalues
eigenvalue_score = (eigenvals.sum(-1) / eigenvals.abs().sum(-1)).real
imshow(eigenvalue_score)
# %%
logits, cache = model.run_with_cache("When John and Mary went to the store, John gave a bottle of milk to")
unembed_dir = model.W_U[:, model.to_single_token(" Mary")]
resid_stack, resid_labels = cache.get_full_resid_decomposition(expand_neurons=False, apply_ln=True, pos_slice=-1, return_labels=True)
mary_dla = resid_stack @ unembed_dir
line((resid_stack @ unembed_dir).squeeze().T, x=resid_labels, title="DLA of Mary Across Layers")
# %%
scatter(x=mary_dla.squeeze(), y=eggs_dla.squeeze(), color=["H" in lab for lab in resid_labels], hover=resid_labels, xaxis="IOI NNMH Score", yaxis="Eggs Score", title="IOI NNMH Score vs Eggs Score")
# %%
gpt2_small = HookedTransformer.from_pretrained("gpt2-small")
utils.test_prompt("When I went to meet Mr Smith and Mrs Johnson I said how do you do Mr", "Smith", gpt2_small)
utils.test_prompt("When I went to meet Mrs Smith and Mr Johnson I said how do you do Mr", "Johnson", gpt2_small)
utils.test_prompt("When I went to meet Mr Smith and Mrs Johnson I said how do you do Mrs", "Johnson", gpt2_small)
utils.test_prompt("When I went to meet Mrs Smith and Mr Johnson I said how do you do Mrs", "Smith", gpt2_small)
# %%
