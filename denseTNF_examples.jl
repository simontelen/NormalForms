# Examples of how to use the functions in denseTNF.jl.
# Simon Telen 9-6-2020

using MultivariatePolynomials
using DynamicPolynomials
include("denseTNF.jl")

#-------------------------------------------------------------------------------
# Intersecting two conics in the plane [Thesis, Example 3.1.2]
@polyvar x y
f = [7.0+3*x-6*y-4*x^2+2*x*y+5*y^2;-1.0-3*x+14*y-2*x^2+2*x*y-3*y^2] # Define the conics
@time sol = solveDense(f,[x;y]) # there are four real solutions

#-------------------------------------------------------------------------------
# Intersecting two conics in the plane with singular intersections [Thesis, Example 3.1.6]
@polyvar x y
f = [x+1/3*y^2-x^2,-1/3*x+1/3*x^2] # Define the conics
@time sol = solveDense(f,[x;y];clustering = true) # the option clustering = true uses reordered schur and the trace for computing singular solutions

#-------------------------------------------------------------------------------
# Another singular example [Master's thesis, p 78]
@polyvar x y
f = [-4+5*x-3*x^2+x^3+5*y-2*x*y-3*y^2+y^3,-4+x-2*x^2+2*x^3+9*y+2*x*y-4*x^2*y-8*y^2+3*x*y^2+y^3] # Define the conics
# if we try to solve this system without clustering, some of the computed solutions are not very accurate:
@time sol = solveDense(f,[x;y])
maximum(get_residual(f,sol,[x;y]))

# There are three solutions to this system, one is simple, one has μ = 3 and one has μ = 5.
# Because of the multiplicity 5, the results are best for applying clustering with a relatively high clustering tolerance (1e-3)
@time sol_clustered = solveDense(f,[x;y];clustering = true, tol = 1e-3) # the option clustering = true uses reordered schur and the trace for computing singular solutions
maximum(get_residual(f,sol_clustered,[x;y]))

#-------------------------------------------------------------------------------
# Intersecting generic curves in ℂ²
n = 2;
d = [20,20]; # Intersecting a degree 7 and a degree 13 curve
@polyvar x[1:n]

# Construct a generic system
f = fill(1.0+sum(x),n)
for i = 1:n
    mons = monomials((1+sum(x))^d[i])
    f[i] = randn(length(mons))'*mons;
end

# Solve the system using dense TNFs
@time sol_QR = solveDense(f,x) # solveDense uses QR with optimal column pivoting for basis selection by default
@time sol_SVD = solveDense(f,x;basisChoice = "SVD") # The option basisChoice = "SVD" makes sure SVD is used for the basis selection

# Compute the relative backward error on the computed approximations.
maximum(get_residual(f,sol_QR,x))
maximum(get_residual(f,sol_SVD,x))

# Extract the real solutions
realSol_QR = getRealSolutions(sol_QR)
realSol_SVD = getRealSolutions(sol_SVD)

#-------------------------------------------------------------------------------
# Intersecting generic surfaces in ℂ³
n = 3;
d = [3,8,11]; # Intersecting a degree 7 and a degree 13 curve
@polyvar x[1:n]

# Construct a generic system
f = fill(1.0+sum(x),n)
for i = 1:n
    mons = monomials((1+sum(x))^d[i])
    f[i] = randn(length(mons))'*mons;
end

# Solve the system using dense TNFs
@time sol_QR = solveDense(f,x) # solveDense uses QR with optimal column pivoting for basis selection by default
@time sol_SVD = solveDense(f,x;basisChoice = "SVD") # The option basisChoice = "SVD" makes sure SVD is used for the basis selection

# Compute the relative backward error on the computed approximations.
maximum(get_residual(f,sol_QR,x))
maximum(get_residual(f,sol_SVD,x))

#-------------------------------------------------------------------------------
# Computing a symbolic resultant map.
n = 3
@polyvar x[1:n]
d = [2,2,2]
f = fill(sum(x),n)
mons = fill(monomials(sum(x)),n)
nmons = fill(0,n)
b = fill(x,n)
for i = 1:n
    mons[i] = monomials((1+sum(x))^d[i]);
    nmons[i] = length(mons[i]);
    @polyvar a[i,1:nmons[i]]
    b[i] = a
    f[i] = b[i]'*mons[i]
end
Ressym, Σ, σ = getRes_dense_sym(f,x,b)
