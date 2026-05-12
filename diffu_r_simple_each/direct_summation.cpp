#include "direct_summation.h"
#include <cmath>

PrecisionSetting directSummation_diffusionCoefficient_pos(
    const std::vector<PositionSpace>& particles, const PositionSpace& target, size_t index_target
) { //calculate particle
    PrecisionSetting I = 0.0;
    PrecisionSetting R_epsilon_squared = R_epsilon*R_epsilon;
    // PrecisionSetting R_epsilon_squared = 0.01*0.01; //debug cpp
    PrecisionSetting dx, dy, dz, r_kl_squared;
    for (size_t j = 0; j < N_particles; ++j) {
        if (index_target == j) continue;

        dx = target.x - particles[j].x;
        dy = target.y - particles[j].y;
        dz = target.z - particles[j].z;
        r_kl_squared = dx * dx + dy * dy + dz * dz;

        I += r_kl_squared / ((r_kl_squared + R_epsilon_squared) * (r_kl_squared + R_epsilon_squared));
    }
    return I * Rm*Rm * coef_Diffu_parallel_iso;
}

// void directSummationVelDiffusion_deltaij(
//     const std::vector<VelocitySpace>& particles, const VelocitySpace& pk, size_t k, Diffu_tensor_vel& tensor
// ){
//     // Initialize tensor elements to zero
//     for (int i = 0; i < 3; ++i)
//         for (int j = 0; j < 3; ++j)
//             tensor.D[i][j] = 0.0;

//     const size_t N = particles.size();
//     for (size_t l = 0; l < N; ++l) {
//         if (k == l) continue;

//         const auto& pl = particles[l];
//         PrecisionSetting dv[3] = {
//             pk.x - pl.x,
//             pk.y - pl.y,
//             pk.z - pl.z
//         };
//         PrecisionSetting V0_sq = dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2] + v_epsilon*v_epsilon;
//         PrecisionSetting V0 = sqrt(V0_sq);
//         PrecisionSetting V0_cubed = V0_sq * V0;

//         for (int i = 0; i < 3; ++i) {
//             for (int j = 0; j < 3; ++j) {
//                 PrecisionSetting delta_ij = (i == j) ? 1.0 : 0.0;
//                 tensor.D[i][j] += (delta_ij / V0 - dv[i]*dv[j] / V0_cubed);
//             }
//         }
//     }

//     // multiply constant
//     for (int i = 0; i < 3; ++i)
//         for (int j = 0; j < 3; ++j){
//             tensor.D[i][j] *= v2m_sqrt*coef_Diffu_tensor_uniform;
//         }
// }

void directSummationVelDiffusion_fraction(
    const std::vector<VelocitySpace>& particles, const VelocitySpace& pk, 
    size_t targetIndex, Diffu_tensor_vel& J_tensor
){
    // Initialize tensor elements to zero
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            J_tensor.D[i][j] = 0.0;

    const size_t N = particles.size();
    
    // Accumulate J_ij contributions
    for (size_t l = 0; l < N; ++l) {
        if (l == targetIndex) continue;

        const VelocitySpace& pl = particles[l];

        PrecisionSetting V0x = pk.x - pl.x;
        PrecisionSetting V0y = pk.y - pl.y;
        PrecisionSetting V0z = pk.z - pl.z;
        PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z + v_epsilon * v_epsilon;
        PrecisionSetting V0 = sqrt(V0_sq);
        PrecisionSetting V0_cubed = V0_sq * V0;

        // Diagonal components J_ii
        PrecisionSetting I11 = (V0y * V0y + V0z * V0z) / V0_cubed;
        PrecisionSetting I22 = (V0z * V0z + V0x * V0x) / V0_cubed;
        PrecisionSetting I33 = (V0x * V0x + V0y * V0y) / V0_cubed;

        // Off-diagonal components J_ij
        PrecisionSetting I12 = (-V0x * V0y) / V0_cubed;
        PrecisionSetting I13 = (-V0x * V0z) / V0_cubed;
        PrecisionSetting I23 = (-V0y * V0z) / V0_cubed;

        J_tensor.D[0][0] += I11;
        J_tensor.D[1][1] += I22;
        J_tensor.D[2][2] += I33;

        J_tensor.D[0][1] += I12;
        J_tensor.D[0][2] += I13;
        J_tensor.D[1][2] += I23;
    }

    // Symmetric assignment
    J_tensor.D[1][0] = J_tensor.D[0][1];
    J_tensor.D[2][0] = J_tensor.D[0][2];
    J_tensor.D[2][1] = J_tensor.D[1][2];

    // Multiply by the external constant
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            J_tensor.D[i][j] *= v2m_sqrt*coef_Diffu_tensor_uniform;
}
