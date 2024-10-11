# Multhithreaded-CPP-Gradient-Descent

## Dataset Preparation
We will be making a simple linear regression model dataset.
So for this we will randomly generate a seed for which we will use std::srand().
### Importance of srand()
->If we don't seed the random number generator, it will produce the same sequence of numbers every time you run the program. This is because random number generators in computers are actually pseudorandom, meaning they use a formula to generate a sequence of numbers that only appears random.
->By setting a seed, you define the starting point for the random number generation. Each different seed will produce a different sequence of numbers.

### Importance of time()
->std::time() returns the current time as the number of seconds since the "epoch" (typically January 1, 1970). When you pass nullptr as an argument, it tells std::time() to simply return the current time rather than store it somewhere.

### Importance of static_cast<unsigned int>(...)
->std::srand() expects an unsigned int as a seed. So, the static_cast<unsigned int>(...) is used to convert the std::time_t value (returned by std::time()) into the correct data type (unsigned int).

So to generate dataset , we use the expression y = theta0 + theta1*x + noise , to make the data like a real life scenario.

for noise we use rand function which will genearte a random number between 0.0 to 0.99 and then scale it to -0.5 to 0.49 to centre it around 0.

And thus we store corresponding X and Y values in vectors and our dataset is prepared.

## Vanilla Gradient Descent

Vanilla gradient descent is a basic optimization algorithm used to minimize a differentiable function by iteratively updating the parameters in the opposite direction of the gradient of the function with respect to those parameters. The goal is to find the set of parameters that minimize the cost function. Each update is controlled by a learning rate, which determines the step size towards the minimum.

We will proceed by 2 way with multi threading without multi threading 
### 1)Without multi threading

![image](https://github.com/user-attachments/assets/447e9e69-50be-4b25-ae05-45430f0ee7d3)


In this hypothesis function function computes the value of h(theta0,theta1) = theta0+theta1*x[i] and the we find error as the difference between hypothesis value and actual value.
The gradient with respect to theta0 is simply the sum of errors.
The gradient w.r.t theta1 is  sum of the product of the errors and the corresponding input feature x[i].
![image](https://github.com/user-attachments/assets/d72c4522-17ae-4b6b-985f-7932f4ec539a)
![image](https://github.com/user-attachments/assets/d9730683-7e26-48b2-aeb2-6197034ab8c6)

then we can simpply update the parameter using update equation
theta(new) = theta(old) + (alpha)*errorTheta{obtained by above code}.

### 2)With Multi-Threading

First create a structure of gradient to store gradient for each thread
For each thread we compute gradient and store it in the struct and as we compute for each thread , at each iteration then we combine the gradient obtained from each thread.

![image](https://github.com/user-attachments/assets/87524641-e77c-4563-ac9f-5249b159bfee)


Dataset is divided into chunk and then each chunk is processed by different chunk and each chunk is processed by a separate thread. Each thread computes the gradient for its assigned portion of the dataset. Once all threads complete their work, the results are combined to update the model parameters.

std::thread(computeGradient, ...): Each thread is created using the std::thread constructor and assigned the computeGradient function to execute. The function takes the following arguments:
1)X and Y: The input feature and target data vectors (passed by constant reference using std::cref to avoid unnecessary copies).
2)theta0 and theta1: The current model parameters (passed as copies to ensure thread safety).
3)start and end: The data range the thread should process.
4)thread_gradients[i]: A reference to a container where the thread will store its computed gradients, allowing the main thread to combine them later.

After that we wait till all the threads combine.
th.joinable(): This checks if the thread is joinable (i.e., if it is still running or can be joined).
th.join(): This blocks the main thread until the thread th finishes execution, ensuring that all threads complete their gradient computations before the main thread proceeds to the next step (like aggregating the gradients and updating the parameters).


Purpose of Multithreading:
Parallelization: Instead of having a single thread process the entire dataset, multiple threads are used to compute the gradients in parallel, speeding up the computation, especially for large datasets.
Work Division: Each thread processes a chunk of the dataset (start to end indices), computes the gradient for that portion, and stores the result in thread_gradients[i].



