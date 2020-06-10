# Functions for solving generic polyhedral systems using truncated normal forms.
# Simon Telen 9-6-2020

using MultivariatePolynomials
using DynamicPolynomials
using LinearAlgebra
using Polymake
using GenericSchur
include("TNFtools.jl")

function getRes_polyhedral(f,x)
    # this computes the toric resultant map similar to Emiris's construction for a square system f
    # it returns also the monomials Σ indexing the columns of res and the shifts σ
    s = length(f)
    n = length(x)
    # This function should probably only be used for s = n
    v = (rand(1:10,n).//rand(1:10,n)).//1000 # random vector to shift the supports
    println("     computing support α + α₀...")
    # The support corresponds to the interior lattice points in the Minkowski sum + a simplex
    f₀ = (randn(n)'*x + randn()) # random linear function
    Π = prod(f)
    prodTerms = terms(Π*f₀)
    latPts = zeros(Rational{Int64}, length(prodTerms), n+1)
    for i = 1:length(prodTerms)
        latPts[i,:] = [1 transpose(exponents(prodTerms[i]) + v)]
    end
    P̂ = @pm polytope.Polytope(POINTS = latPts)
    latPts_α_α₀ = P̂.INTERIOR_LATTICE_POINTS
    latPts_α_α₀ = convert(Array{Int64},latPts_α_α₀[:,2:end])
    Σ_α_α₀ = exptomon(latPts_α_α₀,x)

    # It will be useful to know which lattice points are in the Minkowski sum, without the extra f₀
    println("     computing support α...")
    prodTerms = terms(Π)
    latPts = zeros(Rational{Int64}, length(prodTerms), n+1)
    for i = 1:length(prodTerms)
        latPts[i,:] = [1 transpose(exponents(prodTerms[i]) + v)]
    end
    P = @pm polytope.Polytope(POINTS = latPts)
    latPts_α = P.INTERIOR_LATTICE_POINTS
    latPts_α = convert(Array{Int64},latPts_α[:,2:end])
    Σ_α = exptomon(latPts_α,x)

    # The shifts correspond to the interior lattice points in the Minkowski sum of the other equations + a simplex
    σ = fill([],s)
    for i = 1:s
        println("     computing shifts $i")
        prodTerms = terms(prod(map(k->(k!=i)*f[k] + (k == i)*1,1:s))*f₀)
        latPts = zeros(Rational{Int64}, length(prodTerms), n+1)
        for i = 1:length(prodTerms)
            latPts[i,:] = [1 transpose(exponents(prodTerms[i]) + v)]
        end
        P = @pm polytope.Polytope(POINTS = latPts)
        latPts = P.INTERIOR_LATTICE_POINTS
        latPts = convert(Array{Int64},latPts[:,2:end])
        σ[i] = exptomon(latPts,x)
    end
    return getRes(f,Σ_α_α₀,σ,x), Σ_α_α₀, Σ_α, σ
end

function get_QRbasis_general(N,Σ_α_α₀,Σ_α)
    # This computes an upper triangularized, column pivoted version of N via a pivoted QR factorization of
    # the map N: V → Cᵟ whose columns are indexed by the monomials in Σ_α_α₀.
    # the monomials in the subspace W (largest subspace s.t. W⁺ ⊂ V) are in Σ_α
    mapping = Dict(Σ_α_α₀.=>1:length(Σ_α_α₀))
    δ = size(N,1)
    indsW = map(k->mapping[k],Σ_α)
    indsNotW = findall(i-> i ∉ indsW, 1:length(Σ_α_α₀))
    QRobj = qr(N[:,indsW],Val(true))
    pivots = QRobj.p
    N = triu(QRobj.Q'*N[:,vcat(indsW[pivots],indsNotW)])
    Σ = Σ_α_α₀[vcat(indsW[pivots],indsNotW)]
    return N,Σ
end

function simulDiag_schur(M;clustering = false,tol=1e-4)
    # simultaneous diagonalization of the matrices in M via Schur factorization of a random linear combination.
    # returns an array with the solutions (`third mode` of the CPD)
    solMatrix = simulDiag_schur_mtx(M;clustering = clustering,tol=tol)
    δ = size(solMatrix,1)
    sol = map(i->solMatrix[i,:],1:δ)
    return sol
end

function get_multmtces_QR_general(N,Σ_α_α₀,Σ_α,x)
    # This computes the multiplication matrices via a basis selection which uses QR with optimal column pivoting
    n = length(x)
    Nnew,Σnew = get_QRbasis_general(N,Σ_α_α₀,Σ_α)
    #-------------------------------------------------------------------------
    println("computing multiplication matrices...")
    δ = size(N,1) # number of solutions
    Nb = triu(Nnew[:,1:δ])
    if cond(Nb) > 1e+12
        println("Warning: ill-conditioned. Try homogenization.")
    end
    M = fill(fill(0.0,δ,δ),n) # This will contain the multiplication operators corresponding to the variables
    B = Σnew[1:δ]
    mapping = Dict(Σnew.=>1:length(Σnew))
    for i = 1:n
        indsᵢ = map(k->mapping[x[i]*k],B)
        M[i] = Nb\Nnew[:,indsᵢ];
    end
    return M
end

function get_multmtces_SVD_general(N,Σ_α_α₀,Σ_α,x)
    # This computes the multiplication matrices via a basis selection which uses SVD
    n = length(x)
    mapping = Dict(Σ_α_α₀.=>1:length(Σ_α_α₀))
    δ = size(N,1)
    indsW = map(k->mapping[k],Σ_α)
    indsNotW = findall(i-> i ∉ indsW, 1:length(Σ_α_α₀))
    U,S,V = svd(N[:,indsW])
    Nb = U.*(S[1:δ])'
    if S[δ]/S[1] < 1e-12
        println("Warning: ill-conditioned. Try homogenization.")
    end
    mapping = Dict(Σ_α_α₀.=>1:length(Σ_α_α₀))
    M = fill(fill(0.0,δ,δ),n) # This will contain the multiplication operators corresponding to the variables
    Nbinv = (1 ./(S[1:δ])).*U'
    for i = 1:n
        indsᵢ = map(k->mapping[x[i]*k],Σ_α)
        M[i] = Nbinv*(N[:,indsᵢ]*V[:,1:δ]);
    end
    return M
end

function solvePolyhedral(f,x;basisChoice = "QR", rankTol = 1e3*eps(), clustering = false, tol = 1e-4)
    # Solves a square system f in variables x by using a toric TNF in a monomial basis chosen via QR with column pivoting.
    n = length(x) # number of variables, should be equal to length(f)
    #-------------------------------------------------------------------------
    println("constructing resultant map...")
    @time res, Σ_α_α₀, Σ_α, σ = getRes_polyhedral(f,x)
    println("res is a matrix of size $(size(res))")
    #-------------------------------------------------------------------------
    println("computing cokernel...")
    @time N = transpose(nullspace(transpose(res),rtol = rankTol))
    println("N has size $(size(N))")
    #-------------------------------------------------------------------------
    if basisChoice == "QR"
        println("finding a basis using QR-P...")
        @time M = get_multmtces_QR_general(N,Σ_α_α₀,Σ_α,x)
    elseif basisChoice == "SVD"
        println("finding a basis using SVD...")
        @time M = get_multmtces_SVD_general(N,Σ_α_α₀,Σ_α,x)
    end
    #-------------------------------------------------------------------------
    println("simultaneous diagonalization using Schur factorization...")
    @time sol = simulDiag_schur(M; clustering = clustering, tol = tol)
    return sol
end
