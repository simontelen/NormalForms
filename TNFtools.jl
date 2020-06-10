# Auxiliary functions and struct definitions for the routines in denseTNF.jl, projHNF.jl, 
# polyhedralTNF.jl and toricHNF.jl.

using MultivariatePolynomials
using DynamicPolynomials
using LinearAlgebra
using Polymake
using Statistics
using GenericSchur
using SmithNormalForm  # to add: pkg> add https://github.com/wildart/SmithNormalForm.jl.git

struct regularityPair_info
    Fᵀ # matrix whose rows are primitive ray generators of X
    α # k-vector of integers representing the degree α ∈ Cl(X)
    α₀ # k-vector of integers representing the degree α₀ ∈ Cl(X)
    Σ_α # monomials in n variables corresponding to a ℂ-basis of Γ(X,Oₓ(α))
    Σ_α₀ # monomials in n variables corresponding to a ℂ-basis of Γ(X,Oₓ(α₀))
    Σ_α_α₀ # monomials in n variables corresponding to a ℂ-basis of Γ(X,Oₓ(α + α₀))
end

struct projResult
    affineSolutions # Affine coordinates of the solutions
    homogeneousSolutions # A set of homogeneous coordinates, first coordinate is the homogenization variable
    condition # condition number of the matrix Nb
end

struct toricResult
    toricSolutions # Affine coordinates of the solutions
    homogeneousSolutions # A set of homogeneous coordinates, order corresponds to the order of the ray generators in Fᵀ
    condition # condition number of the matrix Nb
    k # dimension of the Cox ring
    RP::regularityPair_info
end

function getRes(f,Σ,σ,x)
    # this computes the resultant map V₁ × ⋯ × Vₛ → V fast in the monomial bases σ[i] for Vᵢ and Σ for V.
    # x should contain the variables of the system.
    # this assumes real coefficients
    res = fill(0.0,length(Σ),sum(map(i->length(i),σ)))
    n = length(x)
    s = length(f)
    mapping = Dict(Σ.=>1:length(Σ))
    for i = 1:s
        J = sum(map(k->length(k),σ[1:i-1]));
        for j = 1:length(σ[i])
            pol = σ[i][j]*f[i]
            res[map(k->mapping[k],monomials(pol)),J+j] = coefficients(pol)
        end
    end
    return res
end

function getRes_sym(f,Σ,σ,x)
    # this computes the symbolic resultant map V₁ × ⋯ × Vₛ → V in the monomial bases σ[i] for Vᵢ and Σ for V.
    # x should contain the variables of the system.
    # this assumes symbolic coefficients
    res = fill(zero(typeof(x[1])),length(Σ),sum(map(i->length(i),σ)))
    n = length(x)
    s = length(f)
    for i = 1:s
        println("EQUATION $i")
        for j = 1:length(σ[i])
            println("shift $(j) out of $(length(σ[i])).")
            res[:,sum(map(k->length(k),σ[1:i-1])) + j] = map(k->coefficient(σ[i][j]*f[i],k,x),Σ);
        end
    end
    return res
end

function exptomon(M,x)
    # Returns a vector of monomials corresponding to the exponent tuples in the rows of M
    mons = fill(prod(x),size(M,1))
    mtx = x'.^M
    mons = map(k->prod(mtx[k,:]),1:size(mtx,1)) # this gives a vector of monomials
    #mons = (prod(x'.^M, dims = 2)); # this gives a vector of polynomials
end

function get_residual(f, sol::Array{Array{Complex{Float64},1},1}, x)
    # This computes the residuals of the points in sol for the system f in the variables x
    n = length(x)
    residuals = zeros(length(sol))
    for k = 1:length(sol)
        ressol = 0
        for i = 1:length(f)
            R = f[i]
            T = terms(R)
            l = length(T)
            Rabs = sum(abs.([T[s](x=>abs.(sol[k])) for s=1:l]))
            R = R(x=>sol[k])/(1+Rabs)
            ressol = ressol + abs(R)
        end
        residuals[k] = ressol/length(f)
    end
    return residuals
end

function montoexp(Σ)
    # Converts an array of monomials into an array of exponents.
    exparray = exponents.(Σ);
    M = fill(0,length(Σ),length(exparray[1]))
    for i = 1:length(Σ)
        M[i,:] = exparray[i]
    end
    return M
end

function getRealSolutions(sol;tol=1e-10)
    # returns the real solutions in the array sol
    imagNorm = norm.(imag.(sol))./norm.(sol)
    return sol[findall(k-> k<tol,imagNorm)]
end

function solvebinom_SNF(A,λ)
    # notation of `Numerical Root Finding via Cox Rings`
    # solves the binomial system {xᵃ = λⱼ} where a is the j-th column of A.
    F = smith(A)
    k = size(A,1)
    n_α₀ = size(A,2)
    invfactors = F.SNF
    r = findfirst(k->k==0,invfactors)  #rank
    if isnothing(r)
        r = minimum(size(A))
    else
        r = r-1
    end
    U = inv(F.S)
    V = inv(F.T)
    w = transpose(log.(Complex.(λ)))*V[:,1:r]
    w = w*diagm(1 ./ invfactors[1:r])
    println(w)
    logy = vcat(transpose(w),zeros(k-r))
    return transpose(U)*logy
end

function simulDiag_schur_mtx(M;clustering=false, tol=1e-4)
    # simultaneous diagonalization of the matrices in M via Schur factorization of a random linear combination.
    # returns a matrix with the eigenvalues (`third mode` of the CPD)
    n = length(M)
    δ = size(M[1],1)
    Mₕ = sum(randn(n).*M)
    F = schur(Mₕ)
    F = triangularize(F)
    solMatrix = fill(0.0+0.0*im,δ,n)
    if clustering
        oF = ordschur(F,fill(true,δ))
        v = diag(oF.Schur)
        clusvec = getClusters(v;tol=tol)
        push!(clusvec,δ+1)
        Q = oF.Z
        for i = 1:n
            Uᵢ = Q'*M[i]*Q
            eᵢ = fill(zero(ComplexF64),δ)
            for j = 1:length(clusvec)-1
                i₀ = clusvec[j]
                i₁ = clusvec[j+1]-1
                ev = tr(Uᵢ[i₀:i₁,i₀:i₁])/(i₁-i₀+1)
                eᵢ[i₀:i₁] .= ev
            end
            solMatrix[:,i] = eᵢ
        end
    else
        Q = F.Z
        for i = 1:n
            Uᵢ = Q'*M[i]*Q
            if i == 1 && norm(Uᵢ-triu(Uᵢ))>1e-10*norm(Uᵢ)
                println("there may be singular solutions, try the option clustering = true")
            end
            solMatrix[:,i] = diag(Uᵢ)
        end
    end
    return solMatrix
end

function getClusters(v;tol=1e-4)
    # returns a vector clusvec with the indices of the start of each cluster in v.
    # v is a vector of complex number which is assumed to be ordered, e.g. coming from an ordered Schur factorization
    clus = [v[1]]
    clusvec = [1]
    k = 2
    while k <= length(v)
        center = mean(clus)
        if abs(v[k]-center) < tol*(abs(center)+1)
            push!(clus,v[k])
        else
            push!(clusvec,k)
            clus = [v[k]]
        end
        k = k+1
    end
    return clusvec
end

function solvebinom_SNF_array(A,Λ)
    # notation of `Numerical Root Finding via Cox Rings`
    # solves the binomial systems {xᵃ = Λ[i]ⱼ} for i = 1...length(Λ) and a is the j-th column of A.
    F = smith(A)
    k = size(A,1)
    δ = length(Λ)
    n_α₀ = size(A,2)
    invfactors = F.SNF
    r = findfirst(k->k==0,invfactors)  #rank
    if isnothing(r)
        r = minimum(size(A))
    else
        r = r-1
    end
    U = inv(F.S)
    V = inv(F.T)
    Λmat = fill(0.0+0.0*im,length(Λ),n_α₀)
    for i = 1:δ
        Λmat[i,:] = Λ[i]
    end
    w = log.(Complex.(Λmat))*V[:,1:r]
    w = w*diagm(1 ./ invfactors[1:r])
    logY = hcat(w,zeros(δ,k-r))
    logX = logY*U
    loghomsol = map(k->logX[k,:],1:δ)
    homsol = map(k->exp.(loghomsol[k]),1:δ)
    return loghomsol,homsol
end

function computeFan(f̂,x)
    # Computes a facet representation of the Newton polytope of f̂:
    # NP(f̂) = {m | Fᵀm + α ≥ 0 }
    n = length(x)
    prodTerms = terms(prod(f̂))
    latPts = zeros(Int64, length(prodTerms), n+1)
    for i = 1:length(prodTerms)
        latPts[i,:] = [1 transpose(exponents(prodTerms[i]))]
    end
    P = @pm polytope.Polytope(POINTS = latPts)
    aFᵀ = P.FACETS
    Fᵀ = convert(Array{Rational{Int64}},aFᵀ[:,2:end])
    α = convert(Array{Rational{Int64}},aFᵀ[:,1])
    D = denominator.(Fᵀ)
    k = size(Fᵀ,1)
    for i = 1:k
        Fᵀ[i,:] = lcm(D[i,:])*Fᵀ[i,:];
    end
    Fᵀ = convert(Array{Int64}, Fᵀ)
    return Fᵀ , α
end

function classtomonomials(Fᵀ,α,x)
    # computes the monomials in the Cox ring of degree α
    Q_α = @pm polytope.Polytope(FACETS = hcat(α, Fᵀ))
    pts_α = Q_α.LATTICE_POINTS_GENERATORS
    #println(pts_α)
    exps_α = convert(Array{Int64},pts_α[1][:,2:end])
    shift = abs.(minimum(vcat(exps_α,zeros(Int64,1,size(exps_α,2))),dims=1))
    #println(shift)
    exps_α = exps_α .+ shift
    α = - minimum(Fᵀ*transpose(exps_α),dims = 2)
    Σ_α = exptomon(exps_α,x)
    return Σ_α,α
end

function completeRPinfo(Fᵀ,α,α₀,t)
    # completes the ray generators and the degrees α,α₀ to a regularityPair_info
    Σ_α,α = classtomonomials(Fᵀ,α,t)
    Σ_α₀,α₀ = classtomonomials(Fᵀ, α₀,t)
    Σ_α_α₀,β = classtomonomials(Fᵀ,α+α₀,t)
    return regularityPair_info(Fᵀ,α,α₀,Σ_α,Σ_α₀,Σ_α_α₀)
end

function homogenize(f̂,Fᵀ,t,x)
    # homogenize a polynomial f̂ in a subset of the variables t to a polynomial in the variables x in the Ring graded according to Fᵀ
    n = length(t)
    k = length(x)
    mapping = Dict(t.=>1:length(t))
    mons = monomials(f̂)
    expons = zeros(Int64, length(mons), n)
    v = map(ℓ->mapping[ℓ],variables(f̂))
    expons[:,v] = montoexp(monomials(f̂))
    α = - minimum(Fᵀ*transpose(expons),dims = 2)
    α = α[:,1] # turn it into a 1D array
    println(α)
    homexpons = transpose(Fᵀ*transpose(expons) .+ α)
    f = sum(coefficients(f̂) .* exptomon(homexpons,x))
    return f,α
end

function homogenizeSystem(f̂,Fᵀ,t,x)
    dummy = 0.0+sum(x)
    f = fill(dummy,length(f̂))
    k = length(x) # should be equal to size(Fᵀ,1)
    αs = fill(zeros(Int64,k),length(f̂))
    for i = 1:length(f̂)
        f[i],αs[i] = homogenize(f̂[i],Fᵀ,t,x)
    end
    return f, αs
end
