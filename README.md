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
