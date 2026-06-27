#ifndef TREE_METHOD_H
#define TREE_METHOD_H

#include "diffu.h"
#include <vector>
#include <memory>

class Octree {
public:
    struct Node {
        float x_min, x_max, y_min, y_max, z_min, z_max;
        std::vector<int> particle_indices;
        std::vector<std::shared_ptr<Node>> children;

        float center_x = 0.0f, center_y = 0.0f, center_z = 0.0f; // Node's center
        size_t total_particles = 0;                              // Particles count

        Node(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax)
            : x_min(xmin), x_max(xmax), y_min(ymin), y_max(ymax), z_min(zmin), z_max(zmax) {}
    };

    void build(const std::vector<PositionSpace>& particles);
    float calculateDiffusionCoefficient(const PositionSpace& p);

private:
    std::shared_ptr<Node> root;
    const std::vector<PositionSpace>* particles_ptr = nullptr; // Pointer to particles data
    const size_t buildRecursive_size = 20; // Maximum particles in a leaf node
    const float theta = 0.5; // Opening angle threshold for distant nodes

    void buildRecursive(std::shared_ptr<Node> node, size_t depth);
    void calculateRecursive(const std::shared_ptr<Node>& node, const PositionSpace& p, float& result, size_t& depth_recursive_calculate);
    bool isFarEnough(const std::shared_ptr<Node>& node, const PositionSpace& p);
};

#endif