#Exercise 2:
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 2, 4, 4, 6]

sum_x = 0
sum_y = 0
sum_xy = 0
sum_x_squared = 0
count = 0
x_list = []
y_list = []

colors = ['y', 'r', 'g', 'orange', 'b']
labels = ['iteration 1', 'iteration 2', 'iteration 3', 'iteration 4', 'iteration 5']
for i in range(len(x)):
    sum_x += x[i]
    sum_y += y[i]
    sum_xy += x[i] * y[i]
    sum_x_squared += x[i]**2
    count += 1
    x_list.append(x[i])
    y_list.append(y[i])

    small_number = 1e-10
    denominator = (sum_x_squared / count) - (sum_x / count)**2
    if abs(denominator) < small_number:
        w1 = 0
        w0 = sum_y / count
    else:
        w1 = (sum_xy / count - (sum_x / count) * (sum_y / count)) / denominator
        w0 = (sum_y / count) - w1 * (sum_x / count)

    model = [w0 + w1 * x for x in x_list]

    print(f'Iteration {i + 1} Slope: {w1:.2f}, y-intercept: {w0:.2f}')

    if w1 == 0:
        plt.plot(x, [w0 for i in range(len(x))] , c = colors[i], label = labels[i])
    else:
        plt.plot(x_list, model, c = colors[i], label = labels[i])
    plt.scatter(x[i], y[i], c = colors[i])
plt.title('Regresion Line as Points are Added')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.show()