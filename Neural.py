import random

input_values = [1, 2, 3, 4, 5, 6, 7, 8]
goals = [5, 6, 7, 8, 9, 10, 11, 12]
learning_rate = 0.01

n = int(input("Insert the number of hidden neurons: "))

hidden_weights = []
hidden_biases = []

for i in range(n):
    hidden_weights.append(random.randint(-10, 10) / 10)
    hidden_biases.append(random.randint(-10, 10) / 10)

output_weights = []

for i in range(n):
    output_weights.append(random.randint(-10, 10) / 10)

output_bias = random.randint(-10, 10) / 10


def relu(x):
    if x <= 0:
        return 0
    else:
        return x


for epoch in range(10000):

    total_loss = 0

    for e in range(len(input_values)):
        hidden_outputs = []
        hidden_z = []

        for h in range(n):
            z = input_values[e] * hidden_weights[h] + hidden_biases[h]
            hidden_z.append(z)
            hidden_outputs.append(relu(z))

        output = 0

        for h in range(n):
            output += hidden_outputs[h] * output_weights[h]

        output += output_bias

        error = output - goals[e]
        loss = error ** 2
        total_loss += loss

        output_gradient = 2 * error

        old_output_weights = output_weights.copy()

        for h in range(n):
            output_weights[h] -= (
                learning_rate
                * output_gradient
                * hidden_outputs[h]
            )

        output_bias -= learning_rate * output_gradient

        for h in range(n):

            if hidden_z[h] > 0:
                relu_gradient = 1
            else:
                relu_gradient = 0

            hidden_gradient = (
                output_gradient
                * old_output_weights[h]
                * relu_gradient
            )

            hidden_weights[h] -= (
                learning_rate
                * hidden_gradient
                * input_values[e]
            )

            hidden_biases[h] -= (
                learning_rate
                * hidden_gradient
            )

    average_loss = total_loss / len(input_values)

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}, "
            f"Average Loss: {average_loss:.6f}"
        )

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