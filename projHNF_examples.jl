# Examples of how to use the functions in projHNF.jl.
# Simon Telen 9-6-2020

using MultivariatePolynomials
using DynamicPolynomials
include("projHNF.jl")

#-------------------------------------------------------------------------------
# Intersecting two conics in ℙ² ([Thesis, Example 3.2.3])
@polyvar y[1:2]
f̂ = [y[1]^2-3*y[1]*y[2]+2*y[2]^2+1,y[1]^2-y[2]^2-3*y[2]+1]
# We compute the intersections on ℙ² by homogenizing and then computing a homogeneous normal form.
@time sol = solveProj(f̂,y)
# The first homogeneous coordinate of the solutions corresponds to the homogenization variable.
homsol = sol.homogeneousSolutions
# One of the solutions has a small first coordinate; this indicates that there is a solution `at infinity'.
# The coordinates in ℂ² are stored in sol.affineSolutions:
affsol = sol.affineSolutions
# The solution at infinity corresponds to the solution with large affine coordinates.

#-------------------------------------------------------------------------------
# Intersecting generic threefolds in ℙ⁴
n = 4;
d = [3,4,3,5];
@polyvar x[1:n]

f = fill(1.0+sum(x),n)
for i = 1:n
    mons = monomials((1+sum(x))^d[i])
    f[i] = randn(length(mons))'*mons;
end

@time sol_QR = solveProj(f,x) # The default uses QR with optimal column pivoting for the basis selection
affsol = sol_QR.affineSolutions
homsol = sol_QR.homogeneousSolutions
maximum(get_residual(f,affsol,x))
sol_QR.condition

@time sol_SVD = solveProj(f,x;basisChoice = "SVD")
affsol = sol_SVD.affineSolutions
homsol = sol_SVD.homogeneousSolutions
maximum(get_residual(f,affsol,x))
sol_SVD.condition

#-------------------------------------------------------------------------------
# We can deal with multiplicities in the same way as for denseTNF.jl, see denseTNF_examples.jl
@polyvar x y
f = [-4+5*x-3*x^2+x^3+5*y-2*x*y-3*y^2+y^3,-4+x-2*x^2+2*x^3+9*y+2*x*y-4*x^2*y-8*y^2+3*x*y^2+y^3] # Define the conics
@time sol_clustered = solveProj(f,[x;y]; basisChoice = "SVD", clustering = true, tol = 1e-3) # the option clustering = true uses reordered schur and the trace for computing singular solutions
maximum(get_residual(f,sol_clustered.affineSolutions,[x;y]))

#-------------------------------------------------------------------------------
# Lines at infinity [Thesis, Example 3.2.4]
n = 3
@polyvar y[1:3]
mons = monomials(prod(map(k->1.0+1.0*y[k],1:3)))
f = fill(1.0+sum(y),n)
for i = 1:n
    f[i] = randn(length(mons))'*mons;
end
# Notice that the warning 'warning: this might not be zero-dimensional in ℙⁿ' is printed
solveProj(f,y)
# In this case, try the toric approach.
