import time

input_values = [1,2,3,4,5,6,7,8,9,10]
goals = [5,10,15,20,25,30,35,40,45,50]
weight = 0.5
bias = 0.0
learning_rate = 0.01


start = time.time()
for i in range(1000):
    totat_loss = 0
    for e in range (len(input_values)):
        output = (input_values[e] * weight) + bias
        error = goals[e] - output
        loss = error ** 2

        weight += learning_rate * error * input_values[e]
        
        bias += learning_rate * error

        totat_loss += loss
    average_loss = totat_loss/len(input_values)

    if average_loss < 0.0000001:
        break
    

    if i % 100 == 0:
        print(f"Epoch {i}, Average Loss: {average_loss:.6f}")
        
end = time.time()

test_input = int(input("Enter a test value: "))
prediction = test_input * weight + bias
print("")
print(f"--------------------------------\n Training took \033[91m{end - start:.6f}\033[0m seconds\n--------------------------------")
print(f"Weight = {weight:.6f}")
print(f"Bias   = {bias:.6f}")
print(f"Prediction for {test_input} is = {prediction:.15f}")
