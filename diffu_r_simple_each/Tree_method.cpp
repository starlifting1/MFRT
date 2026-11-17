#include "Tree_method.h"
#include "encounter_kernels.h"
#include <algorithm>
#include <cmath>
#include <limits>

void Octree::build(const std::vector<PositionSpace>& particles) {
    if (particles.empty()) throw std::invalid_argument("Cannot build an empty tree");
    if (!(theta >= 0.0) || !std::isfinite(theta)) throw std::invalid_argument("Invalid opening angle");
    particles_ptr=&particles;
    double xmin=particles[0].x,xmax=xmin,ymin=particles[0].y,ymax=ymin,zmin=particles[0].z,zmax=zmin;
    for (const auto& p:particles) {
        if (!std::isfinite(p.x+p.y+p.z)) throw std::invalid_argument("Nonfinite particle");
        xmin=std::min(xmin,p.x); xmax=std::max(xmax,p.x);
        ymin=std::min(ymin,p.y); ymax=std::max(ymax,p.y);
        zmin=std::min(zmin,p.z); zmax=std::max(zmax,p.z);
    }
    root=std::make_shared<Node>(xmin,xmax,ymin,ymax,zmin,zmax);
    root->particle_indices.reserve(particles.size());
    for (size_t i=0;i<particles.size();++i) root->particle_indices.push_back(i);
    buildRecursive(root,0);
}

void Octree::buildRecursive(std::shared_ptr<Node> node,size_t depth) {
    node->total_particles=node->particle_indices.size();
    if (!node->total_particles) return;
    if (node->total_particles<=buildRecursive_size || depth>=32 ||
        (node->x_min==node->x_max && node->y_min==node->y_max && node->z_min==node->z_max)) {
        for (size_t i:node->particle_indices) {
            const auto& p=(*particles_ptr)[i];
            node->center_x+=p.x; node->center_y+=p.y; node->center_z+=p.z;
        }
        node->center_x/=node->total_particles; node->center_y/=node->total_particles; node->center_z/=node->total_particles;
        cacheOpeningRadius(node);
        return;
    }
    const double xm=(node->x_min+node->x_max)/2,ym=(node->y_min+node->y_max)/2,zm=(node->z_min+node->z_max)/2;
    std::vector<std::vector<int>> groups(8);
    for (size_t i:node->particle_indices) {
        const auto& p=(*particles_ptr)[i];
        groups[(p.x>xm)|((p.y>ym)<<1)|((p.z>zm)<<2)].push_back(i);
    }
    std::vector<int>().swap(node->particle_indices);
    for (int k=0;k<8;++k) {
        if (groups[k].empty()) continue;
        auto child=std::make_shared<Node>((k&1)?xm:node->x_min,(k&1)?node->x_max:xm,
            (k&2)?ym:node->y_min,(k&2)?node->y_max:ym,(k&4)?zm:node->z_min,(k&4)?node->z_max:zm);
        child->particle_indices=std::move(groups[k]);
        buildRecursive(child,depth+1);
        node->center_x+=child->center_x*child->total_particles;
        node->center_y+=child->center_y*child->total_particles;
        node->center_z+=child->center_z*child->total_particles;
        node->children.push_back(std::move(child));
    }
    node->center_x/=node->total_particles; node->center_y/=node->total_particles; node->center_z/=node->total_particles;
    cacheOpeningRadius(node);
}

bool Octree::isFarEnough(const std::shared_ptr<Node>& node,const PositionSpace& p) {
    if (theta==0.0) return false;
    // For theta<=1 the distance bound already excludes every point inside
    // the box: |target-COM| <= sqrt(3)*size/2 + COM_offset < size/theta + COM_offset.
    // Keep the explicit test for permissive non-production opening angles.
    if (theta>1.0 && p.x>=node->x_min && p.x<=node->x_max && p.y>=node->y_min && p.y<=node->y_max && p.z>=node->z_min && p.z<=node->z_max) return false;
    const double dx=p.x-node->center_x,dy=p.y-node->center_y,dz=p.z-node->center_z;
    return dx*dx+dy*dy+dz*dz>node->opening_radius_squared;
}

PrecisionSetting Octree::calculateDiffusionCoefficient(const PositionSpace& p) {
    double result=0.0; size_t visited=0;
    calculateRecursive(root,p,result,visited);
    return result*coef_Diffu_pos_trace;
}
void Octree::calculateRecursive(const std::shared_ptr<Node>& node,const PositionSpace& p,double& result,size_t& visited) {
    if (!node || !node->total_particles) return;
    if (isFarEnough(node,p)) {
        const double x=p.x-node->center_x,y=p.y-node->center_y,z=p.z-node->center_z;
        result+=node->total_particles*position_kernel(x*x+y*y+z*z,R_epsilon*R_epsilon);
        return;
    }
    for (size_t i:node->particle_indices) {
        const auto& q=(*particles_ptr)[i];
        if (&p==&q) continue;
        const double x=p.x-q.x,y=p.y-q.y,z=p.z-q.z;
        result+=position_kernel(x*x+y*y+z*z,R_epsilon*R_epsilon);
    }
    ++visited;
    for (const auto& child:node->children) calculateRecursive(child,p,result,visited);
}

void Octree::cacheOpeningRadius(const std::shared_ptr<Node>& node) {
    if (theta==0.0) {node->opening_radius_squared=std::numeric_limits<double>::infinity();return;}
    const double size=std::max({node->x_max-node->x_min,node->y_max-node->y_min,node->z_max-node->z_min});
    const double ox=node->center_x-(node->x_min+node->x_max)/2;
    const double oy=node->center_y-(node->y_min+node->y_max)/2;
    const double oz=node->center_z-(node->z_min+node->z_max)/2;
    const double limit=size/theta+std::sqrt(ox*ox+oy*oy+oz*oz);
    node->opening_radius_squared=limit*limit;
}
