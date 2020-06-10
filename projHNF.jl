# Functions for solving generic total degree systems in ℙⁿ using homogeneous normal forms. 
# Simon Telen 9-6-2020

using MultivariatePolynomials
using DynamicPolynomials
using LinearAlgebra
using GenericSchur
include("TNFtools.jl")

struct projResult
    affineSolutions # Affine coordinates of the solution
    homogeneousSolutions # A set of homogeneous coordinates, first coordinate is the homogenization variable
    condition # condition number of the matrix Nb
end

function getRes_dense(f,x)
    # this computes the dense resultant map similar to Macaulay's construction for a square system f
    # it returns also the monomials Σ indexing the columns of res and the shifts σ
    # compute the total degree of the equations
    s = length(f)
    n = length(x)
    # This function should probably only be used for s = n
    d = fill(0,s)
    for i = 1:s
        d[i] = maximum(degree.(terms(f[i])))
    end
    ρ = sum(d) - n + 1
    Σ = monomials((1+sum(x))^ρ) # support of the resultant map;
    Σ = map(k->k,Σ) # this is to turn Σ into an array. Otherwise the monomials are automatically sorted according to some term order.
    σ = fill([],s)
    for i = 1:s
        σ[i] = monomials((1+sum(x))^(ρ-d[i]))
    end
    return getRes(f,Σ,σ,x),Σ,σ
end

function get_multmtces_SVD_proj(N,Σ,x)
    # This computes homogeneous multiplication matrices in a basis chosen via SVD of
    # the map N: Sᵨ → Cᵟ whose columns are indexed by the (non-homogeneous) monomials in Σ.
    δ = size(N,1)
    n = length(x)
    maxdeg = maximum(degree.(Σ))
    indsW = findall(i-> i < maxdeg ,degree.(Σ))
    mapping = Dict(Σ.=>1:length(Σ))
    Nw = N[:,indsW]
    Σw = Σ[indsW]
    Nᵢ = fill(fill(0.0,size(Nw)),n+1)
    Nᵢ[1] = Nw;
    for i = 2:n+1
        indsᵢ = map(k->mapping[k*x[i-1]],Σw)
        Nᵢ[i] = N[:,indsᵢ]
    end
    Nh₀ = sum(randn(n+1).*Nᵢ); # Dehomogenization with respect to a generic linear form.
    U,S,V = svd(Nh₀)
    κ = S[1]/S[end]
    if κ > 1e12
        println("warning: this might not be zero-dimensional in ℙⁿ")
    end
    Nbinv = (1 ./S[1:δ]).*U';
    M = fill(fill(0.0,δ,δ),n+1) # This will contain the multiplication operators corresponding to the variables
    for i = 1:n+1
        M[i] = Nbinv*(Nᵢ[i]*V[:,1:δ]);
    end
    return M, κ
end

function simulDiag_schur_proj(M;clustering = false,tol=1e-4)
    # simultaneous diagonalization of the matrices in M via Schur factorization of a random linear combination.
    # returns an array with the solutions (`third mode` of the CPD)
    solMatrix = simulDiag_schur_mtx(M;clustering = clustering,tol=tol)
    δ = size(solMatrix,1)
    homsol = map(i->solMatrix[i,1:end],1:δ)
    solMatrix = solMatrix./solMatrix[:,1]
    sol = map(i->solMatrix[i,2:end],1:δ)
    return sol, homsol
end

function get_multmtces_QR_proj(N,Σ,x)
    # This computes homogeneous multiplication matrices in a basis chosen via QR with pivoting of
    # the map N: Sᵨ → Cᵟ whose columns are indexed by the (non-homogeneous) monomials in Σ.
    δ = size(N,1)
    n = length(x)
    maxdeg = maximum(degree.(Σ))
    indsW = findall(i-> i < maxdeg ,degree.(Σ))
    mapping = Dict(Σ.=>1:length(Σ))
    Nw = N[:,indsW]
    Σw = Σ[indsW]
    Nᵢ = fill(fill(0.0,size(Nw)),n+1)
    Nᵢ[1] = Nw;
    for i = 2:n+1
        indsᵢ = map(k->mapping[k*x[i-1]],Σw)
        Nᵢ[i] = N[:,indsᵢ]
    end
    Nh₀ = sum(randn(n+1).*Nᵢ); # Dehomogenization with respect to a generic linear form.
    QRobj = qr(Nh₀,Val(true))
    pivots = QRobj.p
    Nb = triu(QRobj.Q'*Nh₀[:,pivots[1:δ]])
    κ = cond(Nb)
    if κ > 1e12
        println("warning: this might not be zero-dimensional in ℙⁿ")
    end
    M = fill(fill(0.0,δ,δ),n+1) # This will contain the multiplication operators corresponding to the variables
    for i = 1:n+1
        M[i] = Nb\(QRobj.Q'*Nᵢ[i][:,pivots[1:δ]]);
    end
    return M, κ
end

function solveProj(f,x;basisChoice = "QR",clustering = false,tol=1e-4)
    # Solves a square system f in variables x by using a dense TNF in a monomial basis chosen via QR with column pivoting (default)
    # or in a basis coming from SVD.
    n = length(x) # number of variables, should be equal to length(f)
    #-------------------------------------------------------------------------
    println("constructing resultant map...")
    @time res,Σ,σ = getRes_dense(f,x)
    #imshow(log10.(abs.(res)))
    #sleep(10)
    println("res is a matrix of size $(size(res))")
    #-------------------------------------------------------------------------
    println("computing cokernel...")
    @time N = transpose(nullspace(transpose(res)))
    println("N has size $(size(N))")
    #-------------------------------------------------------------------------
    if basisChoice == "QR"
        println("finding a basis using QR-P...")
        @time M, κ = get_multmtces_QR_proj(N,Σ,x)
    elseif basisChoice == "SVD"
        println("finding a basis using SVD...")
        @time M, κ = get_multmtces_SVD_proj(N,Σ,x)
    end
    #println(norm(M[1]*M[2]-M[2]*M[1]))
    #-------------------------------------------------------------------------
    println("simultaneous diagonalization using Schur factorization...")
    @time sol, homsol = simulDiag_schur_proj(M;clustering = clustering,tol=tol)
    return projResult(sol,homsol,κ)
end
