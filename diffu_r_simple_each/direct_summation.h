#ifndef DIRECTSUMMATION_VEL_H
#define DIRECTSUMMATION_VEL_H

#include <vector>
#include "diffu.h"

/*  The target is one sample point as input argument.
*/
PrecisionSetting directSummation_diffusionCoefficient_pos(const std::vector<PositionSpace>& particles, const PositionSpace& target, size_t index_target);

// /*  The target is one sample point as input argument. 
//     This is the oirginal formula of BT08 Appendix L.
// */
// void directSummationVelDiffusion_deltaij(
//     const std::vector<VelocitySpace>& particles, const VelocitySpace& pk, size_t k, Diffu_tensor_vel& tensor
// );

/*  The target is one sample point as input argument. 
    This is might more numerical stable than void directSummationVelDiffusion_deltaij().
*/
void directSummationVelDiffusion_fraction(
    const std::vector<VelocitySpace>& particles, const VelocitySpace& pk, size_t k, Diffu_tensor_vel& tensor
);

#endif // DIRECTSUMMATION_VEL_H
