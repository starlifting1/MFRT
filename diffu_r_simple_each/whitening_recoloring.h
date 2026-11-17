#ifndef WHITENING_RECOLORING_H
#define WHITENING_RECOLORING_H

#include <vector>
#include <Eigen/Dense>
#include "diffu.h"

//struct for velocity dispersion tensor
struct VelocitySampleInfo{
    PrecisionSetting vm_radius, v2m_sqrt;
    std::vector<PrecisionSetting> vmean_3; //vmean_3 = {0., 0., 0.};
    Eigen::Matrix3d dispersion_mat; //dispersion = Eigen::Matrix3d::Zero();
};

void print_VelocitySampleInfo(const std::vector<VelocitySpace>& velocities, const std::string& descrp="descrp");

/* 
   Calculate velocity dispersion from a velocity sample.
   Here the dispersion is defined as the covariance matrix of the velocity components,
   i.e. 
   sigma^2_{ij} = (1/N) * sum_k ((v_k,i - <v_i>)(v_k,j - <v_j>)).
   And return the $\sqrt{ \sigma^2_{ij} }$.
*/
Eigen::Matrix3d calculate_velocity_dispersion_sqrt(const std::vector<VelocitySpace>& velocities);

/**
 * Adjusts a set of velocity samples so that their first and second moments match the target.
 * It is whitening and recoloring.
 *
 * The input sample (velocities) is modified in-place such that:
 *   new_v = A * v + b,
 * where
 *   A = Sigma_target^(1/2) * Sigma_orig^(-1/2)   and   b = targetMean - A * mu_orig.
 *
 * @param velocities A vector of VelocitySpace samples (will be updated).
 * @param targetMean An Eigen::Vector3d containing the desired mean velocity.
 * @param targetCov  An Eigen::Matrix3d containing the desired covariance (second moment about the mean).
 */
void readjust_velocities_dispersion_affine(
    std::vector<VelocitySpace>& velocities, const Eigen::Vector3d& targetMean, const Eigen::Matrix3d& targetCov
);

#endif // WHITENING_RECOLORING_H
