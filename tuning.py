from ngram import NGram

# Tuning parameters - the lower the output perplexity, the better the parameter.
# For K, it seems like the lower the K, the better the resultant perplexity.
tri_model = NGram(ngram_size = 3, vocab_file = 'vocab.txt')
tri_model.train('./train_data')
k_vals = [0.01,0.1,0.5]
lambda_lists= [[0.1,0.2,0.7],[0.2,0.3,0.5],[0.3,0.3,0.4]]
output_perplexity_k = []
output_perplexity_v = []
with open('test_data/test.txt', 'r') as f:
    data = f.read()
for k in k_vals:
    output_k = tri_model.perplexity(data, params = {'k': k})
    output_perplexity_k.append(output_k)
for lambda_list in lambda_lists:
    output_lambda = tri_model.perplexity(data, params = {'lambdas': lambda_list})
    output_perplexity_v.append(output_lambda)

print("The best perplexity for k is: ", round(min(output_perplexity_k), 5))
print("The best perplexity for lambdas is: ", round(min(output_perplexity_v), 5))
