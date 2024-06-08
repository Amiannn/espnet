import torch
import optimized_transducer

N, T, U, D, V = 2, 5, 3, 10, 5

blank_id = 0

encoder_out = torch.rand(N, T, D) # from the encoder
decoder_out = torch.rand(N, U, D) # from the decoder, i.e., the prediction network

targets = torch.randint(0, V, (N, U - 1)).int()

logit_lengths  = torch.IntTensor([T, T]).int()
target_lengths = torch.IntTensor([U - 1, U - 1]).int()
print(f'logit_lengths : {logit_lengths}')
print(f'target_lengths: {target_lengths}')

encoder_out_list = [encoder_out[i, :logit_lengths[i], :] for i in range(N)]
decoder_out_list = [decoder_out[i, :target_lengths[i]+1, :] for i in range(N)]

print(f'encoder_out_list: {[out.shape for out in encoder_out_list]}')
print(f'decoder_out_list: {[out.shape for out in decoder_out_list]}')

x = [e.unsqueeze(1) + d.unsqueeze(0) for e, d in zip(encoder_out_list, decoder_out_list)]
x = [p.reshape(-1, D) for p in x]

for i in x:
    print(f'shape: {i.shape}')
x = torch.cat(x)
print(f'x shape: {x.shape}')
f = torch.nn.Linear(D, V)

activation = torch.tanh(x)
logits = f(activation) # linear is an instance of `nn.Linear`.
print(f'logits shape: {logits.shape}')
loss = optimized_transducer.transducer_loss(
    logits=logits,
    targets=targets,
    logit_lengths=logit_lengths,
    target_lengths=target_lengths,
    blank=blank_id,
    reduction="mean",
    from_log_softmax=False,
)

print(f'loss: {loss}')