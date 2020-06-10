# Examples of how to use the functions in toricHNF.jl.
# Simon Telen 10-6-2020

using MultivariatePolynomials
using DynamicPolynomials
using LinearAlgebra
using PyPlot
import PyPlot: pygui
PyPlot.pygui(true)
include("TNFtools.jl")
include("toricHNF.jl")
include("denseTNF.jl")

#----------------------------------------------------------------------------
# Intersecting curves on a Hirzebruch surface. This is the running example in [Tel].
n = 2
@polyvar t[1:n]
f̂ = [1+t[1]+t[2]+t[1]*t[2]+t[1]^2*t[2]+t[1]^3*t[2]; 1+t[2]+t[1]*t[2]+t[1]^2*t[2]]
Fᵀ,α = computeFan(f̂,t)
α₀ = [1;0;0;0]
RP = completeRPinfo(Fᵀ,α,α₀,t)
R = solveToric(f̂,t,RP)
maximum(get_residual(f̂,R.toricSolutions,t))

#----------------------------------------------------------------------------
#Computing 27 lines on a cubic surface
@polyvar c[1:20] p q r
mon = monomials((1+p+q+r)^3);
hyp = c'*mon;
@polyvar t[1:4]
randcoeffs = randn(20);
randcoeffs = randcoeffs/norm(randcoeffs)
hyp = subs(hyp, c=> randcoeffs, p=> t[1] +t[2]*r, q=> t[3]+t[4]*r);
dummy = 1.0 + sum(t);
f̂ = fill(dummy,4);
for i = 1:4
   f̂[i] = subs(coefficient(hyp, r^(i-1), [r]), r=>1.0);
end

Fᵀ,α = computeFan(f̂,t)
α₀ = [1;0;1;0;0;0]
RP = completeRPinfo(Fᵀ,α,α₀,t)
R = solveToric(f̂,t,RP;torictol = 1e-5,clustering=true,tol = 1e-3)
maximum(get_residual(f̂,R.toricSolutions,t))
semilogy(get_residual(f̂,R.toricSolutions,t))
imshow(log10.(abs.(hcat(R.homogeneousSolutions...))))

# If we don't want to compute the homogeneous coordinates on the boundary, we can set torictol = 0 and still obtain accurate results for the 27 toric solutions.
R = solveToric(f̂,t,RP;torictol = 0,clustering=true,tol = 1e-3)
maximum(get_residual(f̂,R.toricSolutions,t))
semilogy(get_residual(f̂,R.toricSolutions,t))
imshow(log10.(abs.(hcat(R.homogeneousSolutions...))))

# Using the improved regularity bounds from [BT], the computations can be done much faster.
α = [4;0;4;0;0;0]
α₀ = [1;0;1;0;0;0]
RP = completeRPinfo(Fᵀ,α,α₀,t)
R = solveToric(f̂,t,RP;torictol = 1e-5,clustering=true,tol = 1e-3)
maximum(get_residual(f̂,R.toricSolutions,t))
semilogy(get_residual(f̂,R.toricSolutions,t))
imshow(log10.(abs.(hcat(R.homogeneousSolutions...))))

#---------------------------------------------------------------------------------------------
# A dense example to compare the computation time with solveDense from denseTNF.jl
n= 2;
d = [20,20];
@polyvar t[1:n]

f̂ = fill(1.0+sum(t),n)
for i = 1:n
    mons = monomials((1+sum(t))^d[i])
    f̂[i] = randn(length(mons))'*mons;
end

Fᵀ,α = computeFan(f̂,t)
α₀ = [1; 0 ;0]
RP = completeRPinfo(Fᵀ,α,α₀,t)

@time R = solveToric(f̂,t,RP)
@time solveDense(f̂,t)
maximum(get_residual(f̂,R.toricSolutions,t))

#-----------------------------------------------------------------------------------------------
# A random 2D example
n = 2
@polyvar t[1:n]
nmons = 15 # number of terms per equation
degrange = 20 # maximal degree in each of the toric variables
f̂ = fill(1.0+sum(t),n)
monmtx = rand(0:degrange,nmons,n)
for i = 1:n
    mons = exptomon(monmtx,t)
    f̂[i] = randn(length(mons))'*mons;
end

Fᵀ,α = computeFan(f̂,t)
α₀ = 7*ones(Int64,size(Fᵀ,1)) # try degree 7⋅[D₁ + … + Dₖ] for α₀
RP = completeRPinfo(Fᵀ,α,α₀,t)
αmat = montoexp(RP.Σ_α)
α₀mat = montoexp(RP.Σ_α₀)
# visualize the monomials of degree α and α₀ as lattice points to check that those of α₀ affinely generate the lattice
plot(αmat[:,1],αmat[:,2],".")
plot(α₀mat[:,1],α₀mat[:,2],".")

R = solveToric(f̂,t,RP;basisChoice = "SVD")
#maximum(get_residual(f̂,R.toricSolutions,t))
semilogy(get_residual(f̂,R.toricSolutions,t))

#-------------------------------------------------------------------------------------------
# A random bihomogeneous example
n = 2;
d = [10,10];
@polyvar t[1:n]

f̂ = fill(1.0+sum(t),n)
for i = 1:n
    mons = monomials(prod(map(k->sum(t[k].^(0:d[i])),1:n)))
    f̂[i] = randn(length(mons))'*mons;
end
Fᵀ,α = computeFan(f̂,t)
α₀ = [1;1;2;2]
RP = completeRPinfo(Fᵀ,α,α₀,t)
R = solveToric(f̂,t,RP;basisChoice = "QR")
maximum(get_residual(f̂,R.toricSolutions,t))
semilogy(get_residual(f̂,R.toricSolutions,t))

# evaluate the accuracy of the computed homogeneous solutions for the homogenized equations
k = R.k
@polyvar x[1:k]
f,αs = homogenizeSystem(f̂,Fᵀ,t,x)
maximum(get_residual(f,R.homogeneousSolutions,x))

#-------------------------------------------------------------------------------------------
# Curves on the double pillow surface [Thesis, Example 5.5.10]
n = 2
@polyvar t[1:2]
f̂ = [t[1]^2*t[2]-t[1]+t[1]*t[2]^2+t[2];2*t[1]^2*t[2]+t[1]-t[1]*t[2]^2-t[2]]
Fᵀ,α = computeFan(f̂,t)
α₀ = [1;1;1;1]
RP = completeRPinfo(Fᵀ,α,α₀,t)
R = solveToric(f̂,t,RP;basisChoice = "QR",clustering=true)
maximum(get_residual(f̂,R.toricSolutions,t))

k = R.k
@polyvar x[1:k]
f,αs = homogenizeSystem(f̂,Fᵀ,t,x)
maximum(get_residual(f,R.homogeneousSolutions,x))
