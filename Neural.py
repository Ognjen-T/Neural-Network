import random

input_values = [1, 2, 3, 4, 5]
goals = [5, 7, 9, 11, 13]

n = int(input("Insert the number of hidden neurons: "))

hidden_weights = []
hidden_biases = []

for i in range(n):
    hidden_weights.append(random.uniform(-1, 1))
    hidden_biases.append(random.uniform(-1, 1))

output_weights = []
for i in range(n):
    output_weights.append(random.uniform(-1, 1))

output_bias = random.uniform(-1, 1)

def relu(x):
    return max(0, x)

for epoch in range(1000):
    total_loss = 0

    for e in range(len(input_values)):
        hidden_outputs = []

        for h in range(n):
            z = input_values[e] * hidden_weights[h] + hidden_biases[h]
            hidden_outputs.append(relu(z))

        output = 0

        for h in range(n):
            output += hidden_outputs[h] * output_weights[h]

        output += output_bias

        error = goals[e] - output
        loss = error ** 2

        total_loss += loss

    average_loss = total_loss / len(input_values)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Average Loss: {average_loss:.6f}")

print("Training finished")

test_input = float(input("Enter a value: "))

hidden_outputs = []

for h in range(n):
    z = test_input * hidden_weights[h] + hidden_biases[h]
    hidden_outputs.append(relu(z))

prediction = 0

for h in range(n):
    prediction += hidden_outputs[h] * output_weights[h]

prediction += output_bias

print("Prediction:", prediction)