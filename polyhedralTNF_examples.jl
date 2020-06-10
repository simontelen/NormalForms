# Examples of how to use the functions in polyhedralTNF.jl.
# Simon Telen 9-6-2020

using MultivariatePolynomials
using DynamicPolynomials
include("polyhedralTNF.jl")

#-------------------------------------------------------------------------------
# A first polyhedral system, [Thesis, Example 5.3.1]
@polyvar t[1:2]
a = randn(3)
b = randn(4)
f = [sum(a.*[1;t[1]^3*t[2];t[1]*t[2]^3]), sum(b.*[1;t[1]^2;t[2]^2;t[1]^2*t[2]^2])]
sol_toric = solvePolyhedral(f,t)
maximum(get_residual(f,sol_toric,t))

#-------------------------------------------------------------------------------
# Systems with `block supports`, [Thesis, Experiment 5.3.1]
n = 4;
d = [1,1,1,1]
@polyvar x[1:n]
f = fill(1.0+sum(x),n)
for i = 1:n
    mons = monomials(prod(map(k->sum(x[k].^(0:d[i])),1:n)))
    f[i] = randn(length(mons))'*mons;
end

numexps = 5;
totaltime = 0;
for i = 1:numexps
    t = @timed sol_toric_SVD = solvePolyhedral(f,x;basisChoice = "SVD")
    global totaltime = totaltime + t[2]
end
avgtime = totaltime/numexps

@time sol_toric_SVD = solvePolyhedral(f,x;basisChoice = "SVD")
residuals = get_residual(f,sol_toric_SVD,x)
maximum(residuals)
mean(log10.(residuals))
minimum(residuals)

#-------------------------------------------------------------------------------
# Computing molecular configurations following, see [Thesis, Experiment 5.3.2]
@polyvar t[1:3]
#β = [-310 959 774 1389 1313; -365 755 917 1451 269; -413 837 838 1655 1352]
β = [-13 -1 -1 24 -1; -13 -1 -1 24 -1; -13 -1 -1 24 -1]

mons1 = [1 t[2]^2 t[3]^2 t[2]*t[3] t[2]^2*t[3]^2]
mons2 = [1 t[3]^2 t[1]^2 t[3]*t[1] t[3]^2*t[1]^2]
mons3 = [1 t[1]^2 t[2]^2 t[1]*t[2] t[1]^2*t[2]^2]

f = [β[1,:]'*mons1';β[2,:]'*mons2';β[3,:]'*mons3']
@time sol_toric_SVD = solvePolyhedral(f,t;basisChoice = "SVD")
residuals = get_residual(f,sol_toric_SVD,t)
maximum(residuals)
mean(log10.(residuals))
minimum(residuals)
realsol = getRealSolutions(sol_toric_SVD)

#-------------------------------------------------------------------------------
# Random systems with a polyhedral structure
n = 3
@polyvar x[1:n]
nmons = 10 # number of monomials in each equation
degrange = 7 # degree in each variable is bounded by degrange
f = fill(1.0+sum(x),n)
for i = 1:n
    monmtx = rand(0:degrange,nmons,n)
    mons = exptomon(monmtx,x)
    f[i] = randn(length(mons))'*mons;
end
@time sol_toric = solvePolyhedral(f,x)
@time sol_toric_SVD = solvePolyhedral(f,x;basisChoice = "SVD",tol =1e2*eps())
maximum(get_residual(f,sol_toric,x))
maximum(get_residual(f,sol_toric_SVD,x))
