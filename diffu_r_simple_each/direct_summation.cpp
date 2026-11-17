#include "direct_summation.h"
#include "encounter_kernels.h"

PrecisionSetting directSummation_diffusionCoefficient_pos(const std::vector<PositionSpace>& particles,
                                                          const PositionSpace& target,size_t index_target) {
    double result=0.0;
    for (size_t j=0;j<particles.size();++j) {
        if (j==index_target) continue;
        const double x=target.x-particles[j].x,y=target.y-particles[j].y,z=target.z-particles[j].z;
        result+=position_kernel(x*x+y*y+z*z,R_epsilon*R_epsilon);
    }
    return result*coef_Diffu_pos_trace;
}
void directSummationVelDiffusion_fraction(const std::vector<VelocitySpace>& particles,
                                         const VelocitySpace& target,size_t index_target,Diffu_tensor_vel& result) {
    initialize_one_tensor(result);
    for (size_t j=0;j<particles.size();++j) {
        if (j==index_target) continue;
        add_velocity_kernel(result,target.x-particles[j].x,target.y-particles[j].y,target.z-particles[j].z,v_epsilon*v_epsilon);
    }
    result.D[1][0]=result.D[0][1]; result.D[2][0]=result.D[0][2]; result.D[2][1]=result.D[1][2];
    for (auto& row:result.D) for (double& value:row) value*=v2m_sqrt*coef_Diffu_tensor_uniform;
}
