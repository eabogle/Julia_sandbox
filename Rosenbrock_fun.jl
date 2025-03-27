using Optim

# Define the Rosenbrock function
"""
    rosen(x)

The Rosenbrock function, a common test problem for optimization algorithms. 

# Arguments
- `x`: A vector of length 2 representing the input values.

# Returns
- `Float`: The value of the Rosenbrock function evaluated at `x`.
"""
function rosen(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

# Initial guess for optimization
x0 = [0.0, 0.0]

# Perform optimization using default method
result = optimize(rosen, x0)

# Display the optimization result
println("Optimization Result (using default):")
@show result
println("Optimal value: ", result.minimum)
println("Optimal point: ", result.minimizer)
println("--------------------------------------------------------")

# Perform optimization using LBFGS with autodiff
autodiff_result = optimize(rosen, x0, LBFGS(); autodiff = :forward)

# Display the optimization result using autodiff
println("Optimization Result (using autodiff):")
@show autodiff_result
println("Optimal value: ", autodiff_result.minimum)
println("Optimal point: ", autodiff_result.minimizer)
println("--------------------------------------------------------")

