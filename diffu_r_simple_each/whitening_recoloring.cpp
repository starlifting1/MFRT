#include "whitening_recoloring.h"
#include <iostream>
#include <stdexcept>

Eigen::Matrix3d calculate_velocity_dispersion_sqrt(const std::vector<VelocitySpace>& velocities){
    size_t N = velocities.size();
    if (N == 0) {
        std::cerr << "Empty velocity sample.\n";
        return Eigen::Matrix3d::Zero();
    }
    
    // Compute the mean vector.
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& v : velocities) {
        mean(0) += v.x;
        mean(1) += v.y;
        mean(2) += v.z;
    }
    mean /= static_cast<PrecisionSetting>(N);
    // std::cout<<"vel_mean: "<<mean<<"\n";
    
    // Compute the covariance (dispersion) matrix.
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& v : velocities) {
        Eigen::Vector3d diff(v.x - mean(0), v.y - mean(1), v.z - mean(2));
        cov += diff * diff.transpose();
    }
    cov /= static_cast<PrecisionSetting>(N);  // or use (N-1) for sample variance
    
    for(Eigen::Index i=0;i<3;++i){ //to return its sqrt
        for(Eigen::Index j=0;j<3;++j){
            cov(i,j) = std::sqrt( cov(i,j) ); //might be less than zero for too less value
        }
    }
    return cov;
}

void print_VelocitySampleInfo(const std::vector<VelocitySpace>& velocities, const std::string& descrp){
    VelocitySampleInfo VSI;
    VSI.vm_radius = get_mean_radius(velocities);
    VSI.v2m_sqrt = calculate_quadratic_mean(velocities);
    VSI.vmean_3 = get_center(velocities);
    VSI.dispersion_mat = calculate_velocity_dispersion_sqrt(velocities);

    std::cout<<"VelocitySampleInfo of `"<<descrp<<"':\n";
    std::cout<<"vel[0]: "<<velocities[0].x<<" "<<velocities[0].y<<" "<<velocities[0].z<<"\n";
    std::cout<<"vm_radius = "<<VSI.vm_radius<<"\n";
    std::cout<<"v2m_sqrt = "<<VSI.v2m_sqrt<<"\n";
    std::cout<<"vmean_3 = "<<VSI.vmean_3[0]<<" "<<VSI.vmean_3[1]<<" "<<VSI.vmean_3[2]<<"\n";
    std::cout<<"dispersion_mat = \n"<<VSI.dispersion_mat<<"\n";
}

// Helper: Compute the sample mean from a vector of VelocitySpace.
Eigen::Vector3d computeSampleMean(const std::vector<VelocitySpace>& velocities) {
    Eigen::Vector3d mean(0.0, 0.0, 0.0);
    if (velocities.empty())
        throw std::runtime_error("Velocity sample is empty.");
    for (const auto& v : velocities) {
        mean(0) += v.x;
        mean(1) += v.y;
        mean(2) += v.z;
    }
    mean /= static_cast<PrecisionSetting>(velocities.size());
    return mean;
}

// Helper: Compute the sample covariance matrix from a vector of VelocitySpace,
// given the sample mean.
Eigen::Matrix3d computeSampleCovariance(const std::vector<VelocitySpace>& velocities, const Eigen::Vector3d& mean) {
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    size_t N = velocities.size();
    for (const auto& v : velocities) {
        Eigen::Vector3d diff(v.x - mean(0), v.y - mean(1), v.z - mean(2));
        cov += diff * diff.transpose();
    }
    cov /= static_cast<PrecisionSetting>(N);
    return cov;
}

void readjust_velocities_dispersion_affine(
    std::vector<VelocitySpace>& velocities, const Eigen::Vector3d& targetMean, const Eigen::Matrix3d& targetCov
){
    // Step 1: Compute the current sample mean and covariance.
    Eigen::Vector3d mu_orig = computeSampleMean(velocities);
    Eigen::Matrix3d Sigma_orig = computeSampleCovariance(velocities, mu_orig);

    // Ensure Sigma_orig is invertible.
    if (Sigma_orig.determinant() == 0) {
        throw std::runtime_error("The sample covariance matrix is singular.");
    }

    // Compute the square root and inverse square root of Sigma_orig.
    // For symmetric positive definite matrices, use SelfAdjointEigenSolver.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_orig(Sigma_orig);
    if (es_orig.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed for original covariance.");
    }
    // D_orig: eigenvalues (should be positive)
    Eigen::Vector3d D_orig = es_orig.eigenvalues();
    Eigen::Matrix3d U_orig = es_orig.eigenvectors();
    // Square root: S_orig = U_orig * diag(sqrt(D_orig)) * U_orig^T
    Eigen::Matrix3d D_orig_sqrt = D_orig.cwiseSqrt().asDiagonal();
    Eigen::Matrix3d S_orig = U_orig * D_orig_sqrt * U_orig.transpose();
    // Inverse square root: S_orig_inv = U_orig * diag(1/sqrt(D_orig)) * U_orig^T
    Eigen::Matrix3d D_orig_inv_sqrt = D_orig.cwiseSqrt().cwiseInverse().asDiagonal();
    Eigen::Matrix3d S_orig_inv = U_orig * D_orig_inv_sqrt * U_orig.transpose();

    // Compute the square root of target covariance
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_target(targetCov);
    if (es_target.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed for target covariance.");
    }
    Eigen::Vector3d D_target = es_target.eigenvalues();
    Eigen::Matrix3d U_target = es_target.eigenvectors();
    Eigen::Matrix3d D_target_sqrt = D_target.cwiseSqrt().asDiagonal();
    Eigen::Matrix3d S_target = U_target * D_target_sqrt * U_target.transpose();

    // Compute transformation matrix: A = S_target * S_orig_inv.
    Eigen::Matrix3d A = S_target * S_orig_inv;
    // Compute translation vector: b = targetMean - A * mu_orig.
    Eigen::Vector3d b = targetMean - A * mu_orig;

    // Apply the affine transformation to each velocity sample.
    for (auto& v : velocities) {
        Eigen::Vector3d vec(v.x, v.y, v.z);
        Eigen::Vector3d vec_new = A * vec + b;
        v.x = vec_new(0);
        v.y = vec_new(1);
        v.z = vec_new(2);
    }
    
    std::cout << "Whitening and recoloring complete." << std::endl;
}
