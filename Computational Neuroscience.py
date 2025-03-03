input_1 = 0.05
input_2 = 0.10

w1 = 0.15
w2 = 0.20
w3 = 0.25
w4 = 0.30
w5 = 0.40
w6 = 0.45
w7 = 0.50
w8 = 0.55

b1 = 0.35
b2 = 0.60

target_1 = 0.01
target_2 = 0.99

# Forward Propagation
net_h1 = input_1 * w1 + input_2 * w2 + b1
out_h1 = 1 / (1 + (2.718 ** -net_h1))

net_h2 = input_1 * w3 + input_2 * w4 + b1
out_h2 = 1 / (1 + (2.718 ** -net_h2))

net_o1 = out_h1 * w5 + out_h2 * w6 + b2
out_o1 = 1 / (1 + (2.718 ** -net_o1))

net_o2 = out_h1 * w7 + out_h2 * w8 + b2
out_o2 = 1 / (1 + (2.718 ** -net_o2))

total_error_o1 = 0.5 * (target_1 - out_o1) ** 2
total_error_o2 = 0.5 * (target_2 - out_o2) ** 2
total_error = total_error_o1 + total_error_o2

# Backpropagation 
dE_total_o1 = -(target_1 - out_o1)
dE_total_o2 = -(target_2 - out_o2)

dout_o1 = out_o1 * (1 - out_o1)
dout_o2 = out_o2 * (1 - out_o2)

dnet_o1_w5 = out_h1
dnet_o1_w6 = out_h2

dnet_o2_w7 = out_h1
dnet_o2_w8 = out_h2

dE_total_w5 = dE_total_o1 * dout_o1 * dnet_o1_w5
dE_total_w6 = dE_total_o1 * dout_o1 * dnet_o1_w6
dE_total_w7 = dE_total_o2 * dout_o2 * dnet_o2_w7
dE_total_w8 = dE_total_o2 * dout_o2 * dnet_o2_w8

eta = 0.5
w5 -= eta * dE_total_w5
w6 -= eta * dE_total_w6
w7 -= eta * dE_total_w7
w8 -= eta * dE_total_w8

dnet_h1 = w5 * dE_total_o1 * dout_o1 + w7 * dE_total_o2 * dout_o2
dnet_h2 = w6 * dE_total_o1 * dout_o1 + w8 * dE_total_o2 * dout_o2

dout_h1 = out_h1 * (1 - out_h1)
dout_h2 = out_h2 * (1 - out_h2)

dE_total_w1 = dnet_h1 * dout_h1 * input_1
dE_total_w2 = dnet_h1 * dout_h1 * input_2
dE_total_w3 = dnet_h2 * dout_h2 * input_1
dE_total_w4 = dnet_h2 * dout_h2 * input_2

w1 -= eta * dE_total_w1
w2 -= eta * dE_total_w2
w3 -= eta * dE_total_w3
w4 -= eta * dE_total_w4

print("Updated Weights:")
print("w1:", w1)
print("w2:", w2)
print("w3:", w3)
print("w4:", w4)
print("w5:", w5)
print("w6:", w6)
print("w7:", w7)
print("w8:", w8)
print("Total Error after update:", total_error)
