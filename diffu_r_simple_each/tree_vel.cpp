#include "tree_vel.h"
#include <cmath>
#include <algorithm>
#include <iostream>

void Octree_vel::build(const std::vector<VelocitySpace>& particles) {
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

void Octree_vel::calculateDiffusionCoefficient(
    const VelocitySpace& p, Diffu_tensor_vel& result
){
    // Initialize result tensor to zero
    initialize_one_tensor(result); //J_ij

    size_t depth_recursive_calculate = 0;
    calculateRecursive(root, p, result, depth_recursive_calculate);
    
    // Symmetric assignment
    result.D[1][0] = result.D[0][1];
    result.D[2][0] = result.D[0][2];
    result.D[2][1] = result.D[1][2];
    // std::cout<<"depth_recursive_calculate: "<<depth_recursive_calculate<<"\n";

    // Multiply by the external constant
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            result.D[i][j] *= v2m_sqrt*coef_Diffu_tensor_uniform;
        }
    }
}



// Use node->center_x
void Octree_vel::buildRecursive(std::shared_ptr<Node> node, size_t depth) {
    if (node->particle_indices.size() <= buildRecursive_size || depth > buildRecursive_size) {
        node->total_particles = node->particle_indices.size();
        
        // Explicitly calculate mean velocity, begin
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
        // Explicitly calculate mean velocity, end

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

    // Explicitly calculate mean velocity, begin
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
    // Explicitly calculate mean velocity, end
}

bool Octree_vel::isFarEnough(const std::shared_ptr<Node>& node, const VelocitySpace& p) {
    PrecisionSetting V0x = p.x - node->center_x;
    PrecisionSetting V0y = p.y - node->center_y;
    PrecisionSetting V0z = p.z - node->center_z;
    // PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z;
    PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z + v_epsilon * v_epsilon;
    PrecisionSetting size_squared = (node->x_max-node->x_min)*(node->x_max-node->x_min);
    return size_squared / V0_sq < theta * theta;
}



void Octree_vel::calculateDiffusionCoefficient_loop(
    const VelocitySpace& p, Diffu_tensor_vel& result
){
    std::stack<std::shared_ptr<Node>> node_stack;
    node_stack.push(root);
    size_t depth_loop = 0;

    while (!node_stack.empty()) {
        auto node = node_stack.top();
        node_stack.pop();

        if (isFarEnough(node, p)) {
            // Do approximate node method
            size_t factor = node->total_particles;  // Multiplicity factor is critical!
            
            // PrecisionSetting V0x = p.x - (node->x_min + node->x_max) / 2;
            // PrecisionSetting V0y = p.y - (node->x_min + node->x_max) / 2;
            // PrecisionSetting V0z = p.z - (node->x_min + node->x_max) / 2;
            PrecisionSetting V0x = p.x - node->center_x;
            PrecisionSetting V0y = p.y - node->center_y;
            PrecisionSetting V0z = p.z - node->center_z;
            PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z + v_epsilon * v_epsilon;
            PrecisionSetting V0 = sqrt(V0_sq);
            PrecisionSetting V0_cubed = V0_sq * V0; //note: variable inv may slow down compution speed
            
            // Diagonal components I_ii
            PrecisionSetting I11 = factor * (V0y * V0y + V0z * V0z) / V0_cubed;
            PrecisionSetting I22 = factor * (V0z * V0z + V0x * V0x) / V0_cubed;
            PrecisionSetting I33 = factor * (V0x * V0x + V0y * V0y) / V0_cubed;

            // Off-diagonal components I_ij
            PrecisionSetting I12 = factor * (-V0x * V0y) / V0_cubed;
            PrecisionSetting I13 = factor * (-V0x * V0z) / V0_cubed;
            PrecisionSetting I23 = factor * (-V0y * V0z) / V0_cubed;

            result.D[0][0] += I11;
            result.D[1][1] += I22;
            result.D[2][2] += I33;

            result.D[0][1] += I12;
            result.D[0][2] += I13;
            result.D[1][2] += I23;
            
            continue;
        }

        if (!node->children.empty()) {
            for (auto& child : node->children) {
                if (child) node_stack.push(child);
            }
        } else {
            for (size_t index : node->particle_indices) {
                // Do exact summation
                const auto& q = (*particles_ptr)[index];
                if (&p == &q) continue;

                PrecisionSetting V0x = p.x - q.x;
                PrecisionSetting V0y = p.y - q.y;
                PrecisionSetting V0z = p.z - q.z;
                PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z + v_epsilon * v_epsilon;
                PrecisionSetting V0 = sqrt(V0_sq);
                PrecisionSetting V0_cubed = V0_sq * V0;

                // Diagonal components I_ii
                PrecisionSetting I11 = (V0y * V0y + V0z * V0z) / V0_cubed;
                PrecisionSetting I22 = (V0z * V0z + V0x * V0x) / V0_cubed;
                PrecisionSetting I33 = (V0x * V0x + V0y * V0y) / V0_cubed;

                // Off-diagonal components I_ij
                PrecisionSetting I12 = (-V0x * V0y) / V0_cubed;
                PrecisionSetting I13 = (-V0x * V0z) / V0_cubed;
                PrecisionSetting I23 = (-V0y * V0z) / V0_cubed;

                result.D[0][0] += I11;
                result.D[1][1] += I22;
                result.D[2][2] += I33;

                result.D[0][1] += I12;
                result.D[0][2] += I13;
                result.D[1][2] += I23;
            }
        }
        depth_loop++;
    }

    // Symmetric assignment
    result.D[1][0] = result.D[0][1];
    result.D[2][0] = result.D[0][2];
    result.D[2][1] = result.D[1][2];
    // std::cout<<"depth_loop: "<<depth_loop<<"\n";

    // Multiply by the external constant
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            result.D[i][j] *= v2m_sqrt*coef_Diffu_tensor_uniform;
        }
    }
}



void Octree_vel::calculateRecursive(
    const std::shared_ptr<Node>& node, const VelocitySpace& p, Diffu_tensor_vel& result, 
    size_t& depth_recursive_calculate
){
    if (node->particle_indices.empty() && node->children.empty()) return;

    if (isFarEnough(node, p)) { //case of further node, further
        size_t factor = node->total_particles;  // Multiplicity factor is critical!
        
        // PrecisionSetting V0x = p.x - (node->x_min + node->x_max) / 2;
        // PrecisionSetting V0y = p.y - (node->x_min + node->x_max) / 2;
        // PrecisionSetting V0z = p.z - (node->x_min + node->x_max) / 2;
        PrecisionSetting V0x = p.x - node->center_x;
        PrecisionSetting V0y = p.y - node->center_y;
        PrecisionSetting V0z = p.z - node->center_z;
        PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z + v_epsilon * v_epsilon;
        PrecisionSetting V0 = sqrt(V0_sq);
        PrecisionSetting V0_cubed = V0_sq * V0; //note: variable inv may slow down compution speed
        
        // Diagonal components I_ii
        PrecisionSetting I11 = factor * (V0y * V0y + V0z * V0z) / V0_cubed;
        PrecisionSetting I22 = factor * (V0z * V0z + V0x * V0x) / V0_cubed;
        PrecisionSetting I33 = factor * (V0x * V0x + V0y * V0y) / V0_cubed;

        // Off-diagonal components I_ij
        PrecisionSetting I12 = factor * (-V0x * V0y) / V0_cubed;
        PrecisionSetting I13 = factor * (-V0x * V0z) / V0_cubed;
        PrecisionSetting I23 = factor * (-V0y * V0z) / V0_cubed;

        result.D[0][0] += I11;
        result.D[1][1] += I22;
        result.D[2][2] += I33;

        result.D[0][1] += I12;
        result.D[0][2] += I13;
        result.D[1][2] += I23;
        return;
    }

    if (!node->children.empty()) { //case of not near node, more local
        depth_recursive_calculate++;
        for (const auto& child : node->children) {
            calculateRecursive(child, p, result, depth_recursive_calculate);
        }
    } else { //case of final near node, local
        for (size_t index : node->particle_indices) {
            const auto& q = (*particles_ptr)[index];
            if (&p == &q) continue;

            PrecisionSetting V0x = p.x - q.x;
            PrecisionSetting V0y = p.y - q.y;
            PrecisionSetting V0z = p.z - q.z;
            PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z + v_epsilon * v_epsilon;
            PrecisionSetting V0 = sqrt(V0_sq);
            PrecisionSetting V0_cubed = V0_sq * V0;

            // Diagonal components I_ii
            PrecisionSetting I11 = (V0y * V0y + V0z * V0z) / V0_cubed;
            PrecisionSetting I22 = (V0z * V0z + V0x * V0x) / V0_cubed;
            PrecisionSetting I33 = (V0x * V0x + V0y * V0y) / V0_cubed;

            // Off-diagonal components I_ij
            PrecisionSetting I12 = (-V0x * V0y) / V0_cubed;
            PrecisionSetting I13 = (-V0x * V0z) / V0_cubed;
            PrecisionSetting I23 = (-V0y * V0z) / V0_cubed;

            result.D[0][0] += I11;
            result.D[1][1] += I22;
            result.D[2][2] += I33;

            result.D[0][1] += I12;
            result.D[0][2] += I13;
            result.D[1][2] += I23;
        }
    }
}



/*
// Use node->x_min and node->x_max
void Octree_vel::buildRecursive(std::shared_ptr<Node> node, size_t depth) {
    if (node->particle_indices.size() <= buildRecursive_size || depth > buildRecursive_size) {
        node->total_particles = node->particle_indices.size();
        
        // // Explicitly calculate mean velocity, begin
        // node->center_x = node->center_y = node->center_z = 0.0;
        // for (size_t index : node->particle_indices) {
        //     const auto& p = (*particles_ptr)[index];
        //     node->center_x += p.x;
        //     node->center_y += p.y;
        //     node->center_z += p.z;
        // }
        // PrecisionSetting inv_size = 1.0 / node->total_particles;
        // node->center_x *= inv_size;
        // node->center_y *= inv_size;
        // node->center_z *= inv_size;
        // // Explicitly calculate mean velocity, end

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

    // // Explicitly calculate mean velocity, begin
    // node->particle_indices.clear();
    // node->total_particles = 0;
    // node->center_x = node->center_y = node->center_z = 0.0;

    // for (auto& child : node->children) {
    //     buildRecursive(child, depth + 1);
    //     node->total_particles += child->total_particles;
    //     node->center_x += child->center_x * child->total_particles;
    //     node->center_y += child->center_y * child->total_particles;
    //     node->center_z += child->center_z * child->total_particles;
    // }

    // PrecisionSetting inv_total = 1.0 / node->total_particles;
    // node->center_x *= inv_total;
    // node->center_y *= inv_total;
    // node->center_z *= inv_total;
    // // Explicitly calculate mean velocity, end
}

bool Octree_vel::isFarEnough(const std::shared_ptr<Node>& node, const VelocitySpace& p) {
    PrecisionSetting V0x = p.x - node->center_x;
    PrecisionSetting V0y = p.y - node->center_y;
    PrecisionSetting V0z = p.z - node->center_z;
    PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z;
    PrecisionSetting size_squared = (node->x_max-node->x_min)*(node->x_max-node->x_min);
    return size_squared / V0_sq < theta * theta;
}

void Octree_vel::calculateRecursive(
    const std::shared_ptr<Node>& node, const VelocitySpace& p, Diffu_tensor_vel& result, 
    size_t& depth_recursive_calculate
) {
    if (node->particle_indices.empty() && node->children.empty()) return;

    if (isFarEnough(node, p)) {
        size_t factor = node->total_particles;  // Multiplicity factor is critical!
        
        PrecisionSetting V0x = p.x - (node->x_min + node->x_max) / 2;
        PrecisionSetting V0y = p.y - (node->x_min + node->x_max) / 2;
        PrecisionSetting V0z = p.z - (node->x_min + node->x_max) / 2;
        // PrecisionSetting V0x = p.x - node->center_x;
        // PrecisionSetting V0y = p.y - node->center_y;
        // PrecisionSetting V0z = p.z - node->center_z;
        PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z + v_epsilon * v_epsilon;
        PrecisionSetting V0 = sqrt(V0_sq);
        PrecisionSetting V0_cubed = V0_sq * V0;

        // Diagonal components I_ii
        PrecisionSetting I11 = factor * (V0y * V0y + V0z * V0z) / V0_cubed;
        PrecisionSetting I22 = factor * (V0z * V0z + V0x * V0x) / V0_cubed;
        PrecisionSetting I33 = factor * (V0x * V0x + V0y * V0y) / V0_cubed;

        // Off-diagonal components I_ij
        PrecisionSetting I12 = factor * (-V0x * V0y) / V0_cubed;
        PrecisionSetting I13 = factor * (-V0x * V0z) / V0_cubed;
        PrecisionSetting I23 = factor * (-V0y * V0z) / V0_cubed;

        result.D[0][0] += I11;
        result.D[1][1] += I22;
        result.D[2][2] += I33;

        result.D[0][1] += I12;
        result.D[0][2] += I13;
        result.D[1][2] += I23;
        return;
    }

    if (!node->children.empty()) {
        depth_recursive_calculate++;
        for (const auto& child : node->children) {
            calculateRecursive(child, p, result, depth_recursive_calculate);
        }
    } else {
        for (size_t index : node->particle_indices) {
            const auto& q = (*particles_ptr)[index];
            if (&p == &q) continue;

            PrecisionSetting V0x = p.x - (node->x_min + node->x_max) / 2;
            PrecisionSetting V0y = p.y - (node->x_min + node->x_max) / 2;
            PrecisionSetting V0z = p.z - (node->x_min + node->x_max) / 2;
            // PrecisionSetting V0x = p.x - node->center_x;
            // PrecisionSetting V0y = p.y - node->center_y;
            // PrecisionSetting V0z = p.z - node->center_z;
            PrecisionSetting V0_sq = V0x * V0x + V0y * V0y + V0z * V0z + v_epsilon * v_epsilon;
            PrecisionSetting V0 = sqrt(V0_sq);
            PrecisionSetting V0_cubed = V0_sq * V0;

            size_t factor = node->total_particles;  // Multiplicity factor is critical!

            // Diagonal components I_ii
            PrecisionSetting I11 = (V0y * V0y + V0z * V0z) / V0_cubed;
            PrecisionSetting I22 = (V0z * V0z + V0x * V0x) / V0_cubed;
            PrecisionSetting I33 = (V0x * V0x + V0y * V0y) / V0_cubed;

            // Off-diagonal components I_ij
            PrecisionSetting I12 = (-V0x * V0y) / V0_cubed;
            PrecisionSetting I13 = (-V0x * V0z) / V0_cubed;
            PrecisionSetting I23 = (-V0y * V0z) / V0_cubed;

            result.D[0][0] += I11;
            result.D[1][1] += I22;
            result.D[2][2] += I33;

            result.D[0][1] += I12;
            result.D[0][2] += I13;
            result.D[1][2] += I23;
        }
    }
}
*/
