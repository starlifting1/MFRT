#ifndef TREE_VEL_H
#define TREE_VEL_H

#include "diffu.h"
#include <vector>
#include <memory>
#include <stack>

class Octree_vel {
public:
    struct Node {
        PrecisionSetting x_min, x_max, y_min, y_max, z_min, z_max;
        std::vector<int> particle_indices;
        std::vector<std::shared_ptr<Node>> children;

        PrecisionSetting center_x = 0.0, center_y = 0.0, center_z = 0.0; // Node's center
        size_t total_particles = 0;                              // Particles count

        Node(
            PrecisionSetting xmin, PrecisionSetting xmax, PrecisionSetting ymin, PrecisionSetting ymax, 
            PrecisionSetting zmin, PrecisionSetting zmax
        ): x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax), z_min(zmin), z_max(zmax){}
    };

    void build(const std::vector<VelocitySpace>& particles);

    /* About 5.0 sec for 1e5 particles with depth 700 in a PC; one can choose it for depth less than 1000. */
    void calculateDiffusionCoefficient(const VelocitySpace& p, Diffu_tensor_vel& result);

    /* Sbout 7.5 sec for 1e5 particles with depth 1200 in a PC.  */
    void calculateDiffusionCoefficient_loop(const VelocitySpace& p, Diffu_tensor_vel& result);

private:
    std::shared_ptr<Node> root;
    const std::vector<VelocitySpace>* particles_ptr = nullptr; // Pointer to particles data
    const size_t buildRecursive_size = 20; // Maximum particles in a leaf node
    const PrecisionSetting theta = 0.5; // Opening angle threshold for distant nodes

    void buildRecursive(std::shared_ptr<Node> node, size_t depth);
    bool isFarEnough(const std::shared_ptr<Node>& node, const VelocitySpace& p);
    void calculateRecursive(const std::shared_ptr<Node>& node, const VelocitySpace& p, Diffu_tensor_vel& result, size_t& depth_recursive_calculate);
};

#endif