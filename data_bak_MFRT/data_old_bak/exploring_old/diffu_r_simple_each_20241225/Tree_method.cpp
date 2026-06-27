#include "Tree_method.h"
#include <cmath>
#include <algorithm>
#include <iostream>
void Octree::build(const std::vector<PositionSpace>& particles) {
    particles_ptr = &particles; // Store a pointer to the particles
    float x_min = std::numeric_limits<float>::max();
    float x_max = std::numeric_limits<float>::lowest();
    float y_min = std::numeric_limits<float>::max();
    float y_max = std::numeric_limits<float>::lowest();
    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();

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
void Octree::buildRecursive(std::shared_ptr<Node> node, size_t depth) {
    if (node->particle_indices.size() <= buildRecursive_size || depth > buildRecursive_size) {
        node->total_particles = node->particle_indices.size();
        return;
    }

    float x_mid = (node->x_min + node->x_max) / 2;
    float y_mid = (node->y_min + node->y_max) / 2;
    float z_mid = (node->z_min + node->z_max) / 2;

    node->children.resize(8);
    for (int i = 0; i < 8; ++i) {
        float xmin = (i & 1) ? x_mid : node->x_min;
        float xmax = (i & 1) ? node->x_max : x_mid;
        float ymin = (i & 2) ? y_mid : node->y_min;
        float ymax = (i & 2) ? node->y_max : y_mid;
        float zmin = (i & 4) ? z_mid : node->z_min;
        float zmax = (i & 4) ? node->z_max : z_mid;
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
    float dx = (node->x_min + node->x_max) / 2 - p.x;
    float dy = (node->y_min + node->y_max) / 2 - p.y;
    float dz = (node->z_min + node->z_max) / 2 - p.z;
    float r_squared = dx * dx + dy * dy + dz * dz;
    float size_squared = std::pow(node->x_max - node->x_min, 2);
    return size_squared / r_squared < theta * theta;
}
void Octree::calculateRecursive(
    const std::shared_ptr<Node>& node, const PositionSpace& p, float& result, 
    size_t& depth_recursive_calculate
) {
    if (node->particle_indices.empty() && node->children.empty()) return;

    if (isFarEnough(node, p)) {
        float dx = (node->x_min + node->x_max) / 2 - p.x;
        float dy = (node->y_min + node->y_max) / 2 - p.y;
        float dz = (node->z_min + node->z_max) / 2 - p.z;
        float r_squared = dx * dx + dy * dy + dz * dz;
        float b90_squared = b90 * b90;

        result += r_squared / ((r_squared + b90_squared) * (r_squared + b90_squared)) * node->total_particles;
        return;
    }

    for (size_t index : node->particle_indices) {
        const auto& q = (*particles_ptr)[index]; // Neighbor particle
        if (&p == &q) continue; // Skip self-interaction
        float dx = q.x - p.x;
        float dy = q.y - p.y;
        float dz = q.z - p.z;
        float r_squared = dx * dx + dy * dy + dz * dz;
        float b90_squared = b90 * b90;

        result += r_squared / ((r_squared + b90_squared) * (r_squared + b90_squared));
    }

    depth_recursive_calculate++;
    for (const auto& child : node->children) {
        calculateRecursive(child, p, result, depth_recursive_calculate);
    }
}

float Octree::calculateDiffusionCoefficient(const PositionSpace& p) {
    float result = 0.0f;
    size_t depth_recursive_calculate = 0;
    calculateRecursive(root, p, result, depth_recursive_calculate);
    // std::cout<<"depth_recursive_calculate: "<<depth_recursive_calculate<<"\n";
    return result * coef_const_Diffu;
    // return result * coef_const_Diffu*N_particles/depth_recursive_calculate;
}
