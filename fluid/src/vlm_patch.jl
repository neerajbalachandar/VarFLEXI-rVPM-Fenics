# Keep FLOWVLM robust against colinearity/typing edge cases.

vlm.VLMSolver._regularize(true)

function vlm.VLMSolver._V_AB(A::Vector{<:vlm.VLMSolver.FWrap}, B, C, gamma; ign_col::Bool=false)
    r0 = B - A
    r1 = C - A
    r2 = C - B
    crss = LinearAlgebra.cross(r1, r2)
    magsqr = LinearAlgebra.dot(crss, crss) + (vlm.VLMSolver.regularize ? vlm.VLMSolver.core_rad : 0)

    TF = gamma === nothing ? promote_type(eltype(A), eltype(B), eltype(C)) :
                             promote_type(eltype(A), eltype(B), eltype(C), typeof(gamma))

    if vlm.VLMSolver._check_collinear(magsqr / LinearAlgebra.norm(r0), vlm.VLMSolver.col_crit; ign_col=ign_col)
        return zeros(TF, 3)
    end

    F1 = crss / magsqr
    aux = r1 / sqrt(LinearAlgebra.dot(r1, r1)) - r2 / sqrt(LinearAlgebra.dot(r2, r2))
    F2 = LinearAlgebra.dot(r0, aux)

    if vlm.VLMSolver.blobify
        F1 *= vlm.VLMSolver.gw(LinearAlgebra.norm(crss) / LinearAlgebra.norm(r0), vlm.VLMSolver.smoothing_rad)
    end

    return gamma === nothing ? (F1 * F2) : ((gamma / 4 / pi) * F1 * F2)
end

function vlm.VLMSolver._V_Ainf_out(A::Vector{<:vlm.VLMSolver.FWrap},
                                   infD::Vector{<:vlm.VLMSolver.FWrap}, C, gamma;
                                   ign_col::Bool=false)
    AC = C - A
    unitinfD = infD / sqrt(LinearAlgebra.dot(infD, infD))
    AAp = LinearAlgebra.dot(unitinfD, AC) * unitinfD
    Ap = AAp + A

    boundAAp = vlm.VLMSolver._V_AB(A, Ap, C, gamma; ign_col=ign_col)

    ApC = C - Ap
    crss = LinearAlgebra.cross(infD, ApC)
    mag = sqrt(LinearAlgebra.dot(crss, crss) + (vlm.VLMSolver.regularize ? vlm.VLMSolver.core_rad : 0))

    TF = gamma === nothing ? promote_type(eltype(A), eltype(infD), eltype(C)) :
                             promote_type(eltype(A), eltype(infD), eltype(C), typeof(gamma))

    if vlm.VLMSolver._check_collinear(mag, vlm.VLMSolver.col_crit; ign_col=ign_col)
        return zeros(TF, 3)
    end

    h = mag / sqrt(LinearAlgebra.dot(infD, infD))
    n = crss / mag
    F = n / h

    if vlm.VLMSolver.blobify
        F *= vlm.VLMSolver.gw(h, vlm.VLMSolver.smoothing_rad)
    end

    return gamma === nothing ? (F + boundAAp) : ((gamma / 4 / pi) * F + boundAAp)
end