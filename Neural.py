import time

input_value = int(input("Enter a value: "))
goal = 158641
weight = 0.5
bias = 0.0
learning_rate = 0.01


start = time.time()
for i in range(1000):
    output = (input_value * weight) + bias
    error = goal - output
    loss = (goal - output) ** 2

    if loss < 0.0000001:
        break

    weight += learning_rate * error * input_value

    bias += learning_rate * error
    
end = time.time()
print("")
print(f"--------------------------------\n Training took \033[91m{end - start:.6f}\033[0m seconds\n--------------------------------")
print("Input =", input_value)
print("Output =", output)
print("Error =", error)
print("Loss =", loss)
print("Weight =", weight)