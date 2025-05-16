# inspiration to some extent: https://www.youtube.com/watch?v=AsNTP8Kwu80
# using torch mainly for element-wise operations
import torch

# linear sequence
data = torch.tensor([1., 0.8, 0.6, 0.4])

seq_len = 3
index = 0

x = data[index:index + seq_len]
y = data[index + 1:index + seq_len + 1]

# init
lr = 0.001

loss = [0] * seq_len
relu_output = [0] * seq_len
predictions = [0] * seq_len

# rand init
w_1 = torch.tensor([0.93])
b_1 = torch.tensor([0.64])

w_2 = torch.tensor([0.41])
b_2 = torch.tensor([-0.72])

w_3 = torch.tensor([-0.36])

for z in range(60_000 + 1):
    """forward pass"""
    for i in range(len(x)):
        if i == 0:
            # no previous hidden state
            relu_output[i] = (x[i] * w_1 + b_1).clamp(min=0)
        else:
            relu_output[i] = (x[i] * w_1 + b_1 + w_3 * relu_output[i - 1]).clamp(min=0)

        predictions[i] = relu_output[i] * w_2 + b_2
        loss[i] = (predictions[i] - y[i]) ** 2

        if z % 10000 == 0:
            print(f"Time-step {i}: prediction = {predictions[i].item():.2f}, "
                  f"target = {y[i].item():.2f}, loss = {loss[i].item()}")

    grad_y_hat_0 = 0
    grad_out_0 = 0
    grad_y_hat_1 = 0
    grad_out_1 = 0
    grad_y_hat_2 = 0
    grad_out_2 = 0

    """backward pass of the 0th time-step loss => chain rule = local gradient * grad output"""
    # 0th time-step update
    grad_y_hat_0 += 2 * (predictions[0] - y[0])

    d_relu = 1.0 if relu_output[0] > 0 else 0.0
    grad_out_0 += grad_y_hat_0 * w_2 * d_relu

    """backward pass of the 1st time-step loss => chain rule = local gradient * grad output"""
    # 1st time-step update
    grad_y_hat_1 += 2 * (predictions[1] - y[1])

    d_relu = 1.0 if relu_output[1] > 0 else 0.0
    grad_out_1 += grad_y_hat_1 * w_2 * d_relu

    """backward pass of the 2nd time-step loss => chain rule = local gradient * grad output"""
    # 2nd time-step update
    grad_y_hat_2 += 2 * (predictions[2] - y[2])

    d_relu = 1.0 if relu_output[2] > 0 else 0.0
    grad_out_2 += grad_y_hat_2 * w_2 * d_relu

    # "unrolling" => 2nd / 1st time-step loss is also influenced by some of the 0th and 1st / Oth time-step computations.
    # 1st time-step update
    d_relu = 1.0 if relu_output[1] > 0 else 0.0
    grad_out_1 += d_relu * grad_out_2 * w_3  # MAIN CAUSE OF GRADIENT ISSUE

    # Oth time-step update
    d_relu = 1.0 if relu_output[0] > 0 else 0.0
    grad_out_0 += d_relu * grad_out_1 * w_3  # MAIN CAUSE OF GRADIENT ISSUE

    """update"""
    b_2 -= lr * (grad_y_hat_0 + grad_y_hat_1 + grad_y_hat_2)
    w_2 -= lr * (grad_y_hat_0 * relu_output[0] + grad_y_hat_1 * relu_output[1] + grad_y_hat_2 * relu_output[2])

    b_1 -= lr * (grad_out_0 + grad_out_1 + grad_out_2)
    w_1 -= lr * (grad_out_0 * x[0] + grad_out_1 * x[1] + grad_out_2 * x[2])

    w_3 -= lr * (grad_out_1 * relu_output[0] + grad_out_2 * relu_output[1])

# final parameters
print(w_1, b_1, w_2, b_2, w_3)
