# Functions for solving generic total degree systems using truncated normal forms. 
# Simon Telen 9-6-2020

using MultivariatePolynomials
using DynamicPolynomials
using LinearAlgebra
using GenericSchur
include("TNFtools.jl")

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

function getRes_dense_sym(f,x,b)
    # this computes the dense symbolic resultant map similar to Macaulay's construction for a square system f
    # it returns also the monomials Σ indexing the columns of res and the shifts σ

    # compute the total degree of the equations
    s = length(f)
    n = length(x)
    # This function should probably only be used for s = n
    d = fill(0,s)
    for i = 1:s
        d[i] = maximum(degree.(terms(subs(f[i],b[i]=>ones(length(b[i]))))))
    end
    ρ = sum(d) - n + 1
    Σ = monomials((1+sum(x))^ρ) # support of the resultant map;
    Σ = map(k->k,Σ) # this is to turn Σ into an array. Otherwise the monomials are automatically sorted according to some term order.
    σ = fill([],s)
    for i = 1:s
        σ[i] = monomials((1+sum(x))^(ρ-d[i]))
    end
    return getRes_sym(f,Σ,σ,x),Σ,σ
end


function get_QRbasis_dense(N,Σ)
    # This computes an upper triangularized, column pivoted version of N via a pivoted QR factorization of
    # the map N: V → Cᵟ whose columns are indexed by the monomials in Σ.
    δ = size(N,1)
    maxdeg = maximum(degree.(Σ))
    indsW = findall(i-> i < maxdeg ,degree.(Σ))
    indsNotW = findall(i-> i == maxdeg ,degree.(Σ))
    QRobj = qr(N[:,indsW],Val(true))
    pivots = QRobj.p
    N = (QRobj.Q'*N[:,vcat(indsW[pivots],indsNotW)])
    Σ = Σ[vcat(indsW[pivots],indsNotW)]
    return N,Σ
end

function get_multmtces_SVD_dense(N,Σ,x)
    # This computes an upper triangularized, column pivoted version of N via a pivoted QR factorization of
    # the map N: V → Cᵟ whose columns are indexed by the monomials in Σ.
    n = length(x)
    δ = size(N,1)
    maxdeg = maximum(degree.(Σ))
    indsW = findall(i-> i < maxdeg ,degree.(Σ))
    indsNotW = findall(i-> i == maxdeg ,degree.(Σ))
    U,S,V = svd(N[:,indsW])
    #basis = Σ[indsW]'*V[:,1:δ] # This takes ridiculously long!
    Nb = U.*(S[1:δ])'
    mapping = Dict(Σ.=>1:length(Σ))
    Σw = Σ[indsW];
    M = fill(fill(0.0,δ,δ),n) # This will contain the multiplication operators corresponding to the variables
    Nbinv = (1 ./(S[1:δ])).*U'
    for i = 1:n
        indsᵢ = map(k->mapping[x[i]*k],Σw)
        M[i] = Nbinv*(N[:,indsᵢ]*V[:,1:δ]);
    end
    return M
end

function simulDiag_schur(M;clustering = false,tol=1e-4)
    # simultaneous diagonalization of the matrices in M via Schur factorization of a random linear combination.
    # returns an array with the solutions (`third mode` of the CPD)
    solMatrix = simulDiag_schur_mtx(M;clustering = clustering,tol=tol)
    δ = size(solMatrix,1)
    sol = map(i->solMatrix[i,:],1:δ)
    return sol
end

function get_multmtces_QR_dense(N,Σ,x)
    # Computes the multiplication matrices in a basis picked by QR with pivoting.
    n = length(x)
    Nnew,Σnew = get_QRbasis_dense(N,Σ)
    δ = size(N,1) # number of solutions
    Nb = triu(Nnew[:,1:δ])
    M = fill(fill(0.0,δ,δ),n) # This will contain the multiplication operators corresponding to the variables
    B = Σnew[1:δ]
    mapping = Dict(Σnew.=>1:length(Σnew))
    for i = 1:n
        indsᵢ = map(k->mapping[x[i]*k],B)
        M[i] = Nb\Nnew[:,indsᵢ];
    end
    return M
end

function solveDense(f,x;basisChoice = "QR",clustering = false,tol = 1e-4)
    # Solves a square system f in variables x by using a dense TNF in a monomial basis chosen via QR with column pivoting.
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
        println("finding a basis using QR-P and computing multiplication matrices...")
        @time M = get_multmtces_QR_dense(N,Σ,x)
    elseif basisChoice == "SVD"
        println("finding a basis using SVD and computing multiplication matrices...")
        @time M = get_multmtces_SVD_dense(N,Σ,x)
    end
    #println(norm(M[1]*M[2]-M[2]*M[1]))
    #-------------------------------------------------------------------------
    println("simultaneous diagonalization using Schur factorization...")
    @time sol = simulDiag_schur(M;clustering = clustering,tol = tol)
    return sol
end
