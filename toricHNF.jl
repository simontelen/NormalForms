# Functions for solving polyhedral systems using homogeneous normal forms on the Cox ring of a toric variety.
# Simon Telen 9-6-2020

using MultivariatePolynomials
using DynamicPolynomials
using LinearAlgebra
using Polymake
using GenericSchur
include("TNFtools.jl")

function getRes_toric(f̂,RP::regularityPair_info,t)
    # This computes the graded resultant map of the system f̂ corresponding to the regularity pair RP.
    Fᵀ = RP.Fᵀ
    s = length(f̂)
    n = length(t)
    σ = fill([],s)
    mapping = Dict(t.=>1:length(t))
    for i = 1:s
        trms = terms(f̂[i])
        latPts = zeros(Rational{Int64}, length(trms), n)
        v = map(k->mapping[k],variables(f̂[i]))
        for i = 1:length(trms)
            latPts[i,v] = transpose(exponents(trms[i]))
        end
        αᵢ = -minimum(Fᵀ*latPts', dims = 2)
        Qᵢ = @pm polytope.Polytope(FACETS = hcat(RP.α + RP.α₀ - αᵢ, Fᵀ))
        latPtsQ = Qᵢ.LATTICE_POINTS_GENERATORS
        σ[i] = exptomon(convert(Array{Int64},latPtsQ[1][:,2:end]),t)
    end
    return getRes(f̂,RP.Σ_α_α₀,σ,t),σ
end

function get_multmtces_SVD_toric(N,Σ_α_α₀,Σ_α,Σ_α₀,t)
    # This computes the multiplication matrices via a basis selection which uses SVD
    δ = size(N,1)
    mapping = Dict(Σ_α_α₀.=>1:length(Σ_α_α₀))
    n = length(t)
    n_α₀ = length(Σ_α₀)
    Nᵢ = fill(fill(0.0,δ,length(Σ_α)),n_α₀)
    for i = 1:n_α₀
        indsᵢ = map(k->mapping[k*Σ_α₀[i]],Σ_α)
        Nᵢ[i] = N[:,indsᵢ]
    end
    Nh₀ = sum(randn(n_α₀).*Nᵢ); # Dehomogenization with respect to a generic linear form.
    U,S,V = svd(Nh₀)
    κ = S[1]/S[end]
    if κ > 1e12
        println("warning: this might not be zero-dimensional in X")
    end
    M = fill(fill(0.0,δ,δ),n_α₀) # This will contain the multiplication operators corresponding to the variables
    Nbinv = (1 ./(S[1:δ])).*U'
    for i = 1:n_α₀
        M[i] = Nbinv*(Nᵢ[i]*V[:,1:δ]);
    end
    return M, κ
end

function get_multmtces_QR_toric(N,Σ_α_α₀,Σ_α,Σ_α₀,t)
    # This computes the multiplication matrices via a basis selection which uses QR with optimal column pivoting
    δ = size(N,1)
    mapping = Dict(Σ_α_α₀.=>1:length(Σ_α_α₀))
    n = length(t)
    n_α₀ = length(Σ_α₀)
    Nᵢ = fill(fill(0.0,δ,length(Σ_α)),n_α₀)
    for i = 1:n_α₀
        indsᵢ = map(k->mapping[k*Σ_α₀[i]],Σ_α)
        #println(indsᵢ)
        Nᵢ[i] = N[:,indsᵢ]
    end
    Nh₀ = sum(randn(n_α₀).*Nᵢ); # Dehomogenization with respect to a generic linear form.
    QRobj = qr(Nh₀,Val(true))
    pivots = QRobj.p
    Nb = triu(QRobj.R[1:δ,1:δ])
    κ = cond(Nb)
    if κ > 1e12
        println("warning: this might not be zero-dimensional in X")
    end
    M = fill(fill(0.0,δ,δ),n_α₀) # This will contain the multiplication operators corresponding to the variables
    for i = 1:n_α₀
        M[i] = Nb\(QRobj.Q'*Nᵢ[i][:,pivots[1:δ]]);
    end
    return M, κ
end

#=
function solveBinomial(Σ,Fᵀ,α,λ)
    #solves the binomial system corresponding to Σ[i] = λᵢ, where Σ is a vector of monomials
    logx = solveBinomial_log(Σ,Fᵀ,α,λ)
    return exp.(logx)
end

function solveBinomial_log(Σ,Fᵀ,α,λ)
    M = montoexp(Σ)
    M = convert(Array{Int64,2},((Fᵀ*transpose(M) .+ α)) )#homogenization
    return solvebinom_SNF(M,λ)
end
=#
function solveBinomialSystem_boundary(Σ,Fᵀ,α,λ)
    # Finds one solution of the binomial system given by x^(Fᵀmᵢ+α) = λ_{i}
    # where mᵢ are the exponents of the monomials in Σ. It returns the log of the homogeneous coordinates,
    # the homogeneous coordinates and the toric coordinates of the δ solutions
    # This assumes all of the coordinates in Λ are nonzero.
    M = montoexp(Σ)
    M = convert(Array{Int64,2}, (Fᵀ*transpose(M) .+ α))
    rowsel = findall(k->norm(M[k,:]) > 0,1:size(M,1))
    M = M[rowsel,:]
    k = size(Fᵀ,1)
    loghomsol = fill(-50.0+0.0im,k)
    loghomsol[rowsel] = solvebinom_SNF(M,λ)
    homsol = exp.(loghomsol)
    sol = exp.(transpose(Fᵀ)*loghomsol)
    return sol, homsol, loghomsol
end

function solveBinomialSystems(Σ,Fᵀ,α,Λ)
    # Finds one solution of all binomial systems given by x^(Fᵀmᵢ+α) = λ_{ij}, j = 1 … δ
    # where mᵢ are the exponents of the monomials in Σ. It returns the log of the homogeneous coordinates,
    # the homogeneous coordinates and the toric coordinates of the δ solutions
    # This assumes all of the coordinates in Λ are nonzero.
    M = montoexp(Σ)
    M = convert(Array{Int64,2}, (Fᵀ*transpose(M) .+ α))
    loghomsol,homsol = solvebinom_SNF_array(M,Λ)
    sol = map(k-> exp.(transpose(Fᵀ)*loghomsol[k]),1:length(Λ))
    return sol, homsol, loghomsol
end

function solveBinomialSystems2(Σ,Fᵀ,α,monsol;torictol = 1e-16)
    # Finds one solution of all binomial systems given by x^(Fᵀmᵢ+α) = λ_{ij}, j = 1 … δ
    # where mᵢ are the exponents of the monomials in Σ. It returns the log of the homogeneous coordinates,
    # the homogeneous coordinates and the toric coordinates of the δ solutions
    k = size(Fᵀ,1)
    toricInds = findall(k-> minimum(abs.(k))>norm(k)*torictol,monsol)
    sol, homsol, loghomsol = solveBinomialSystems(Σ,Fᵀ,α,monsol[toricInds])
    boundaryInds = findall(k-> minimum(abs.(k))<=norm(k)*torictol,monsol)
    boundarymonsol = monsol[boundaryInds]
    for i = 1:length(boundarymonsol)
        activeInds = findall(k->abs(k)>torictol*norm(boundarymonsol[i]),boundarymonsol[i])
        tsol, thomsol, tloghomsol = solveBinomialSystem_boundary(Σ[activeInds],Fᵀ,α,boundarymonsol[i][activeInds])
        push!(sol,tsol)
        push!(homsol,thomsol)
        push!(loghomsol,tloghomsol)
    end
    return sol,homsol,loghomsol
end

function simulDiag_schur_toric(M;clustering = false,tol= 1e-4)
    # simultaneous diagonalization of the matrices in M via Schur factorization of a random linear combination.
    # returns an array with the solutions (`third mode` of the CPD)
    solMatrix = simulDiag_schur_mtx(M;clustering = clustering,tol=tol)
    δ = size(solMatrix,1)
    monsol = map(i->solMatrix[i,:],1:δ)
    return monsol
end

function solveToric(f̂,t,RP::regularityPair_info; basisChoice = "QR", clustering = false, tol = 1e-4, torictol = 1e-10)
    n = length(t) # number of toric variables, should be equal to length(f̂)
    #-------------------------------------------------------------------------
    println("constructing resultant map...")
    @time res, σ = getRes_toric(f̂,RP,t)
    println("res is a matrix of size $(size(res))")
    #-------------------------------------------------------------------------
    println("computing cokernel...")
    @time N = transpose(nullspace(transpose(res)))
    println("N has size $(size(N))")
    #-------------------------------------------------------------------------
    if basisChoice == "QR"
        println("finding a basis using QR-P...")
        @time M, κ = get_multmtces_QR_toric(N,RP.Σ_α_α₀,RP.Σ_α,RP.Σ_α₀,t)
    elseif basisChoice == "SVD"
        println("finding a basis using SVD...")
        @time M, κ = get_multmtces_SVD_toric(N,RP.Σ_α_α₀,RP.Σ_α,RP.Σ_α₀,t)
    end
    #println(norm(M[2]*M[1]-M[1]*M[2]))
    #-------------------------------------------------------------------------
    println("simultaneous diagonalization using Schur factorization...")
    @time monsol = simulDiag_schur_toric(M;clustering = clustering,tol=tol)
    #-------------------------------------------------------------------------
    println("solving binomial systems...")
    sol, homsol, loghomsol = solveBinomialSystems2(RP.Σ_α₀,RP.Fᵀ,RP.α₀,monsol;torictol = torictol)
    return toricResult(sol,homsol,κ,size(Fᵀ,1),RP)
end
