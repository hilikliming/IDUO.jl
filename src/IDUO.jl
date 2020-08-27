module IDUO
export iduo, omp
using ProgressMeter
using Base.Threads, Random, SparseArrays, LinearAlgebra
Random.seed!(1234)  # for stability of tests

function error_matrix(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, k::Int)
    # indices = [i for i in 1:size(D, 2) if i != k]
    indices = deleteat!(collect(1:size(D, 2)), k)
    return Y - D[:, indices] * X[indices, :]
end


function init_dictionary(n::Int, K::Int)
    # D must be a full-rank matrix
    D = rand(n, K)
    while rank(D) != min(n, K)
        D = rand(n, K)
    end

    @inbounds for k in 1:K
        D[:, k] ./= norm(@view(D[:, k]))
    end
    return D
end

## JJH 4-24-20 BEGIN
function init_dictionary_data(Data::AbstractMatrix, K::Int)
    # D must be a full-rank matrix
    n=size(Data,1)
    ind1=randperm(size(Data,2))[1:K]
    D = Data[:,ind1]
    while rank(D) != min(n, K)
        ind1=randperm(size(Data,2))[1:K]
        D = Data[:,ind1]
    end

    @inbounds for k in 1:K
        D[:, k] ./= norm(@view(D[:, k]))
        D[:, k] .= D[:,k].*sign(D[1,k]) # multiply in the sign of the first element. JJH
    end
    return D
end

# Performing the Atom updates
function iduo(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, m_atoms::Int)
    N  = size(Y, 2)
    kis= randperm(m_atoms)[1:end] .+ (size(X,1) - m_atoms)
    for k in kis #1:size(X, 1)
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        wₖ = findall(!iszero, xₖ)

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        Eₖ = error_matrix(Y, D, X, k)
        Ωₖ = sparse(wₖ, 1:length(wₖ), ones(length(wₖ)), N, length(wₖ))
        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        U, S, V = svd(Eₖ * Ωₖ, full=true)
        D[:, k] = U[:, 1]
        X[k, wₖ] = V[:, 1] * S[1]
    end
    return D, X
end

function iduo(Y::AbstractMatrix, D_old::AbstractMatrix, m_atoms::Int;
              max_iter::Int = default_max_iter,
              max_iter_mp::Int = default_max_iter_mp, data_init::Bool, des_avg_err::Float64)

    K1       = m_atoms
    Dknews   = IKSVD_atom_init(Y,D_old,max_iter_mp,des_avg_err,m_atoms)
    n, N     = size(Y)
    X_sparse = zeros(1,1)
    Dnew     = zeros(1,1)
    for atom in 1:K1
        display(string("Solving Atom No. ",atom))
        J       = 0
        dKp1    = Dknews[:,atom]
        Dnew    = cat(D_old,dKp1,dims=2)

        # Run Fast OMP on the new data using Dnew to get R and X_Kp1(0)
        X_sparse    = omp(Y,Dnew,max_iter_mp,des_avg_err)
        R           = Y-Dnew*X_sparse
        eps1=1e-8
        ip=1
        Jmax=20
        while ip>eps1 && J<=Jmax
            J=J+1
            Rx=R*X_sparse[end,:]
            Dlast=Dnew[:,end]
            Dnew[:,end]=Rx/sqrt(Rx'*Rx)
            X_sparse[end,:]= R'*Dnew[:,end]
            ip=1-abs(Dlast'*Dnew[:,end])
            Xs=reshape(X_sparse[end,:],1,N)
            ds=reshape(Dnew[:,end],n,1)#convert(Array{Float64,2},Dnew[:,end])
            #R = R-ds*Xs
        end
        D_old=Dnew
    end
    return Dnew, X_sparse

end

function IKSVD_atom_init(Y::AbstractMatrix,D_old::AbstractMatrix,max_iter_mp::Int,des_avg_err::Float64,m_atoms::Int)
    X_sparse    = omp(Y, D_old, max_iter_mp, des_avg_err)
    EY   = Y-D_old*X_sparse
    errs = diag(EY'EY) # Diagonal of gram is the vector of individual samples SqErr in representaiton

    s_errs=sort(errs,rev=true)     # sorted errors of the candidates
    i_errs=sortperm(errs,rev=true) # Indices of the highest error candidates

    max_cand=min(2*m_atoms,length(i_errs))
    i_cand=i_errs[1:max_cand]

    ## Next we compute the entropy of the poorly represented samples
    H   =computeEntropy(X_sparse[:,i_cand])
    i_H =sortperm(H,rev=true)
    ## select m_atoms samples with highest entropy (16) should be an argmax in the paper...
    Dnew = Y[:,i_cand[i_H[1:m_atoms]]]
end

function computeEntropy(X::AbstractMatrix)
    n,N = size(X)
    p   = broadcast(abs,X)
    for ii in 1:N
        p[:,ii]= p[:,ii]/sum(p[:,ii])
    end
    lp = broadcast(log,p)
    H = sum(p.*lp,dims=1)
    return vec(H)
end

# A simple OMP implementation that uses recursive block inversion to solve LS estimates faster
function omp(data::Array{Float64,2}, dictionary::Array{Float64,2}, tau::Int, tolerance::Float64)
    K= size(dictionary,2)
    X_sparse=zeros(K,size(data,2))
    for nn=1:size(data,2)
        x = deepcopy(data[:,nn]) # a single data sample to be represented in terms of atoms in D
        r = deepcopy(x) # residual vector
        D = dictionary # Dictionary
        # Note that the o (optimal) variables are used to avoid an uncommon
        # scenario (that does occur) where a lower sparsity solution may have
        # had lower error than the final solution (with tau non zeros) but
        # wasn't low enough to break out of the coefficient solver via the error
        # tolerance. A litte more memory for significantly better solutions,
        # thanks to CR for the tip (JJH)
        γ       = 0 # this will be the growing coefficient vector
        γₒ      = 0 # this will store whatever the minimum error solution was during computation of the coefficients
        av_err  = 100 # norm of the error vector.
        best_err= 100 # will store lowest error vector norm
        ii      = 1   # while loop index
        DI      = []  # This holds the atoms selected via OMP as its columns (it grows along 2nd dimension)
        DIGI    = []  # Inverse of DI's gram matrix
        DIdag   = []  # PseudoInverse of DI
        I       = []  # set of indices corresponding to atoms selected in reconstruction
        Iₒ      = []  # I think you get the deal with these guys now (best set of indices lul)
        while (length(I)<tau) && (av_err > tolerance)
            k = argmax(broadcast(abs,D'*r))
            dk= D[:,k]
            if ii==1
                I = k
                #display("we made it")
                DI=dk
                DIGI=(DI'*DI)^(-1)
            else
                I = cat(dims=1,I,k)
                rho=DI'*dk
                DI=cat(dims=2,DI,dk)
                ipk=dk'*dk
                DIGI=blockMatrixInv(DIGI,rho,rho,ipk)
            end
            DIdag   = DIGI*DI'
            γ   = DIdag*x
            r       = x-DI*γ
            av_err  = norm(r)
            if av_err<= best_err
                best_err=av_err
                γₒ= γ
                Iₒ=I
            end
            X_sparse[I,nn]=γ
            ii+=1
        end
        if av_err > best_err
            X_sparse[I,nn]= 0*X_sparse[I,nn]
            X_sparse[Iₒ,nn]=γₒ
        end

    end
    return X_sparse
end

function blockMatrixInv(Ai::Array{Float64,2}, B::Array{Float64,1}, C::Array{Float64,1}, D::Float64)
    C=C'
    DCABi= (D-C*Ai*B)^(-1)
    return [Ai+Ai*B*DCABi*C*Ai -Ai*B*DCABi; -DCABi*C*Ai DCABi]
end

function blockMatrixInv(Ai::Float64, B::Float64, C::Float64, D::Float64)
    DCABi= (D-C*Ai*B)^(-1)
    return [Ai+Ai*B*DCABi*C*Ai -Ai*B*DCABi; -DCABi*C*Ai DCABi]
end

end # End Module
