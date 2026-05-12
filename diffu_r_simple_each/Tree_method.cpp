#include "Tree_method.h"
#include <cmath>
#include <algorithm>
#include <iostream>

void Octree::build(const std::vector<PositionSpace>& particles) {
    particles_ptr = &particles; // Store a pointer to the particles
    PrecisionSetting x_min = std::numeric_limits<PrecisionSetting>::max();
    PrecisionSetting x_max = std::numeric_limits<PrecisionSetting>::lowest();
    PrecisionSetting y_min = std::numeric_limits<PrecisionSetting>::max();
    PrecisionSetting y_max = std::numeric_limits<PrecisionSetting>::lowest();
    PrecisionSetting z_min = std::numeric_limits<PrecisionSetting>::max();
    PrecisionSetting z_max = std::numeric_limits<PrecisionSetting>::lowest();

    for (const auto& p : particles) {
        x_min = std::min(x_min, p.x);
        x_max = std::max(x_max, p.x);
        y_min = std::min(y_min, p.y);
        y_max = std::max(y_max, p.y);
        z_min = std::min(z_min, p.z);
        z_max = std::max(z_max, p.z);
    }

    root = std::make_shared<Node>(x_min, x_max, y_min, y_max, z_min, z_max);
    for (size_t i = 0; i < particles.size(); ++i) {
        root->particle_indices.push_back(i);
    }

    buildRecursive(root, 0);
}

PrecisionSetting Octree::calculateDiffusionCoefficient(const PositionSpace& p) {
    PrecisionSetting result = 0.0; //I
    size_t depth_recursive_calculate = 0;
    calculateRecursive(root, p, result, depth_recursive_calculate);
    // std::cout<<"depth_recursive_calculate: "<<depth_recursive_calculate<<"\n";
    // return result * coef_const_Diffu*N_particles/depth_recursive_calculate;
    return result * Rm*Rm * coef_Diffu_parallel_iso;
}






/*
// Use node->center_x
void Octree::buildRecursive(std::shared_ptr<Node> node, size_t depth) {
    if (node->particle_indices.size() <= buildRecursive_size || depth > buildRecursive_size) {
        node->total_particles = node->particle_indices.size();
        
        // Explicitly calculate mean velocity of leaf node
        node->center_x = node->center_y = node->center_z = 0.0;
        for (size_t index : node->particle_indices) {
            const auto& p = (*particles_ptr)[index];
            node->center_x += p.x;
            node->center_y += p.y;
            node->center_z += p.z;
        }
        PrecisionSetting inv_size = 1.0 / node->total_particles;
        node->center_x *= inv_size;
        node->center_y *= inv_size;
        node->center_z *= inv_size;

        return;
    }

    PrecisionSetting x_mid = (node->x_min + node->x_max) / 2;
    PrecisionSetting y_mid = (node->y_min + node->y_max) / 2;
    PrecisionSetting z_mid = (node->z_min + node->z_max) / 2;

    node->children.resize(8);
    for (int i = 0; i < 8; ++i) {
        PrecisionSetting xmin = (i & 1) ? x_mid : node->x_min;
        PrecisionSetting xmax = (i & 1) ? node->x_max : x_mid;
        PrecisionSetting ymin = (i & 2) ? y_mid : node->y_min;
        PrecisionSetting ymax = (i & 2) ? node->y_max : y_mid;
        PrecisionSetting zmin = (i & 4) ? z_mid : node->z_min;
        PrecisionSetting zmax = (i & 4) ? node->z_max : z_mid;
        node->children[i] = std::make_shared<Node>(xmin, xmax, ymin, ymax, zmin, zmax);
    }

    for (size_t index : node->particle_indices) {
        const auto& p = (*particles_ptr)[index];
        size_t child_index = ((p.x > x_mid) << 0) | ((p.y > y_mid) << 1) | ((p.z > z_mid) << 2);
        node->children[child_index]->particle_indices.push_back(index);
    }

    // Explicitly calculate mean velocity
    node->particle_indices.clear();
    node->total_particles = 0;
    node->center_x = node->center_y = node->center_z = 0.0;

    for (auto& child : node->children) {
        buildRecursive(child, depth + 1);
        node->total_particles += child->total_particles;
        node->center_x += child->center_x * child->total_particles;
        node->center_y += child->center_y * child->total_particles;
        node->center_z += child->center_z * child->total_particles;
    }

    PrecisionSetting inv_total = 1.0 / node->total_particles;
    node->center_x *= inv_total;
    node->center_y *= inv_total;
    node->center_z *= inv_total;
}

bool Octree::isFarEnough(const std::shared_ptr<Node>& node, const PositionSpace& p) {
    // PrecisionSetting dx = (node->x_min + node->x_max) / 2 - p.x;
    // PrecisionSetting dy = (node->y_min + node->y_max) / 2 - p.y;
    // PrecisionSetting dz = (node->z_min + node->z_max) / 2 - p.z;
    PrecisionSetting dx = node->center_x - p.x;
    PrecisionSetting dy = node->center_y - p.y;
    PrecisionSetting dz = node->center_z - p.z;
    PrecisionSetting r_squared = dx * dx + dy * dy + dz * dz;
    PrecisionSetting size_squared = std::pow(node->x_max - node->x_min, 2);
    return size_squared / r_squared < theta * theta;
}

void Octree::calculateRecursive(
    const std::shared_ptr<Node>& node, const PositionSpace& p, PrecisionSetting& result, 
    size_t& depth_recursive_calculate
) {
    if (node->particle_indices.empty() && node->children.empty()) return;

    if (isFarEnough(node, p)) {
        // PrecisionSetting dx = (node->x_min + node->x_max) / 2 - p.x;
        // PrecisionSetting dy = (node->y_min + node->y_max) / 2 - p.y;
        // PrecisionSetting dz = (node->z_min + node->z_max) / 2 - p.z;
        PrecisionSetting dx = node->center_x - p.x;
        PrecisionSetting dy = node->center_y - p.y;
        PrecisionSetting dz = node->center_z - p.z;
        PrecisionSetting r_squared = dx * dx + dy * dy + dz * dz;
        PrecisionSetting R_epsilon_squared = R_epsilon * R_epsilon;

        result += r_squared / ((r_squared + R_epsilon_squared) * (r_squared + R_epsilon_squared)) * node->total_particles;
        return;
    }

    for (size_t index : node->particle_indices) {
        const auto& q = (*particles_ptr)[index]; // Neighbor particle
        if (&p == &q) continue; // Skip self-interaction
        PrecisionSetting dx = q.x - p.x;
        PrecisionSetting dy = q.y - p.y;
        PrecisionSetting dz = q.z - p.z;
        PrecisionSetting r_squared = dx * dx + dy * dy + dz * dz;
        PrecisionSetting R_epsilon_squared = R_epsilon * R_epsilon;

        result += r_squared / ((r_squared + R_epsilon_squared) * (r_squared + R_epsilon_squared));
    }

    depth_recursive_calculate++;
    for (const auto& child : node->children) {
        calculateRecursive(child, p, result, depth_recursive_calculate);
    }
}
*/






// /*
// Use node->x_min and node->x_max
void Octree::buildRecursive(std::shared_ptr<Node> node, size_t depth) {
    if (node->particle_indices.size() <= buildRecursive_size || depth > buildRecursive_size) {
        node->total_particles = node->particle_indices.size();
        return;
    }

    PrecisionSetting x_mid = (node->x_min + node->x_max) / 2;
    PrecisionSetting y_mid = (node->y_min + node->y_max) / 2;
    PrecisionSetting z_mid = (node->z_min + node->z_max) / 2;

    node->children.resize(8);
    for (int i = 0; i < 8; ++i) {
        PrecisionSetting xmin = (i & 1) ? x_mid : node->x_min;
        PrecisionSetting xmax = (i & 1) ? node->x_max : x_mid;
        PrecisionSetting ymin = (i & 2) ? y_mid : node->y_min;
        PrecisionSetting ymax = (i & 2) ? node->y_max : y_mid;
        PrecisionSetting zmin = (i & 4) ? z_mid : node->z_min;
        PrecisionSetting zmax = (i & 4) ? node->z_max : z_mid;
        node->children[i] = std::make_shared<Node>(xmin, xmax, ymin, ymax, zmin, zmax);
    }

    for (size_t index : node->particle_indices) {
        const auto& p = (*particles_ptr)[index];
        size_t child_index = ((p.x > x_mid) << 0) | ((p.y > y_mid) << 1) | ((p.z > z_mid) << 2);
        node->children[child_index]->particle_indices.push_back(index);
    }

    node->particle_indices.clear();
    node->total_particles = 0;

    for (auto& child : node->children) {
        buildRecursive(child, depth + 1);
        node->total_particles += child->total_particles;
    }
}

bool Octree::isFarEnough(const std::shared_ptr<Node>& node, const PositionSpace& p) {
    PrecisionSetting dx = (node->x_min + node->x_max) / 2 - p.x;
    PrecisionSetting dy = (node->y_min + node->y_max) / 2 - p.y;
    PrecisionSetting dz = (node->z_min + node->z_max) / 2 - p.z;
    // PrecisionSetting dx = node->center_x - p.x;
    // PrecisionSetting dy = node->center_y - p.y;
    // PrecisionSetting dz = node->center_z - p.z;
    PrecisionSetting r_squared = dx * dx + dy * dy + dz * dz;
    PrecisionSetting size_squared = std::pow(node->x_max - node->x_min, 2);
    return size_squared / r_squared < theta * theta;
}

void Octree::calculateRecursive(
    const std::shared_ptr<Node>& node, const PositionSpace& p, PrecisionSetting& result, 
    size_t& depth_recursive_calculate
) {
    if (node->particle_indices.empty() && node->children.empty()) return;

    if (isFarEnough(node, p)) {
        PrecisionSetting dx = (node->x_min + node->x_max) / 2 - p.x;
        PrecisionSetting dy = (node->y_min + node->y_max) / 2 - p.y;
        PrecisionSetting dz = (node->z_min + node->z_max) / 2 - p.z;
        // PrecisionSetting dx = node->center_x - p.x;
        // PrecisionSetting dy = node->center_y - p.y;
        // PrecisionSetting dz = node->center_z - p.z;
        PrecisionSetting r_squared = dx * dx + dy * dy + dz * dz;
        PrecisionSetting R_epsilon_squared = R_epsilon * R_epsilon;

        result += r_squared / ((r_squared + R_epsilon_squared) * (r_squared + R_epsilon_squared)) * node->total_particles;
        return;
    }

    for (size_t index : node->particle_indices) {
        const auto& q = (*particles_ptr)[index]; // Neighbor particle
        if (&p == &q) continue; // Skip self-interaction
        PrecisionSetting dx = q.x - p.x;
        PrecisionSetting dy = q.y - p.y;
        PrecisionSetting dz = q.z - p.z;
        PrecisionSetting r_squared = dx * dx + dy * dy + dz * dz;
        PrecisionSetting R_epsilon_squared = R_epsilon * R_epsilon;

        result += r_squared / ((r_squared + R_epsilon_squared) * (r_squared + R_epsilon_squared));
    }

    depth_recursive_calculate++;
    for (const auto& child : node->children) {
        calculateRecursive(child, p, result, depth_recursive_calculate);
    }
}
// */
