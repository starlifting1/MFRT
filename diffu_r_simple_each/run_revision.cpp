#include "diffu.h"
#include "generate_samples.h"
#include "direct_summation.h"
#include "Tree_method.h"
#include "tree_vel.h"
#include "fractal_dim.h"
#include "whitening_recoloring.h"
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <limits>

namespace fs = std::filesystem;
// Command-line driver for diffusion traces and cumulative neighbour counts.
struct Options {
    std::string space="pos",kind="uniform",output;
    size_t n=1000,direct=0,count=10000;
    uint64_t seed=20261030,angular_seed=701;
    unsigned perlin_seed=123456;
    int threads=2,iterations=14,octaves=4,nr=12,diagnostic_counts=0;
    double radius=75,mass=137,theta=.5,eps_factor=1,veps_factor=0;
    double scale=-1,step=-1,persistence=.5,lacunarity=2,count_min=-1,count_max=-1;
    std::vector<double> axes{8,6,1};
};
uint64_t unsigned_argument(const std::string& value,const std::string& key) {
    if(value.empty() || !std::all_of(value.begin(),value.end(),[](char c){return c>='0' && c<='9';}))
        throw std::invalid_argument(key+" requires a nonnegative decimal integer");
    return std::stoull(value);
}
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string key=argv[i];
        if(key=="--help") {
            std::cout<<"out_revision.exe --space pos|vel --kind uniform|perlin|radial|iso|aniso|tail|noised|composite --n N --output DIR [--seed S --theta .5 --threads 2 --direct-targets 0 --count-targets 10000 --R0 75 --mass 137 --eps-factor 1 --veps-factor 0 --iterations 14 --noise-scale F --noise-step L --perlin-seed S --angular-seed S --axes 8,6,1 --count-min R --count-max R --count-radii 12]\n";
            std::exit(0);
        }
        if(i+1==argc) throw std::invalid_argument("Missing value for "+key);
        std::string v=argv[++i];
        if(key=="--space") o.space=v; else if(key=="--kind") o.kind=v;
        else if(key=="--output") o.output=v;
        else if(key=="--n") o.n=unsigned_argument(v,key);
        else if(key=="--seed") o.seed=unsigned_argument(v,key);
        else if(key=="--angular-seed") o.angular_seed=unsigned_argument(v,key);
        else if(key=="--perlin-seed") {
            auto seed=unsigned_argument(v,key);
            if(seed>std::numeric_limits<unsigned>::max())throw std::invalid_argument("Perlin seed exceeds 32-bit range");
            o.perlin_seed=seed;
        }
        else if(key=="--threads") o.threads=std::stoi(v);
        else if(key=="--direct-targets") o.direct=unsigned_argument(v,key);
        else if(key=="--count-targets") o.count=unsigned_argument(v,key);
        else if(key=="--R0") o.radius=std::stod(v);
        else if(key=="--mass") o.mass=std::stod(v);
        else if(key=="--theta") o.theta=std::stod(v);
        else if(key=="--eps-factor") o.eps_factor=std::stod(v);
        else if(key=="--veps-factor") o.veps_factor=std::stod(v);
        else if(key=="--iterations") o.iterations=std::stoi(v);
        else if(key=="--noise-scale") o.scale=std::stod(v);
        else if(key=="--noise-step") o.step=std::stod(v);
        else if(key=="--octaves") o.octaves=std::stoi(v);
        else if(key=="--persistence") o.persistence=std::stod(v);
        else if(key=="--lacunarity") o.lacunarity=std::stod(v);
        else if(key=="--count-min") o.count_min=std::stod(v);
        else if(key=="--count-max") o.count_max=std::stod(v);
        else if(key=="--count-radii") o.nr=std::stoi(v);
        else if(key=="--diagnostic-counts") o.diagnostic_counts=std::stoi(v);
        else if(key=="--axes") {
            std::replace(v.begin(),v.end(),',',' ');std::istringstream in(v);
            if(!(in>>o.axes[0]>>o.axes[1]>>o.axes[2])) throw std::invalid_argument("Three axes required");
        } else throw std::invalid_argument("Unknown argument "+key);
    }
    const std::set<std::string> pk{"uniform","perlin","radial"},vk{"iso","aniso","tail","noised","composite"};
    if((o.space!="pos" && o.space!="vel") || !(o.space=="pos"?pk:vk).count(o.kind))
        throw std::invalid_argument("Invalid space or sample kind");
    if(o.output.empty() || o.n<64 || o.n>100000000 || o.threads<1 || o.iterations<0 || o.nr<2 || o.octaves<1)
        throw std::invalid_argument("Invalid output, N, thread count or sampling options");
    // Preserve the dimensionless field when the physical system is rescaled.
    // Explicit --noise-scale / --noise-step remain dimensional overrides.
    const double relative_scale=o.space=="pos" ? o.radius/75. :
        std::sqrt((o.mass/137.)/(o.radius/75.));
    if(o.scale==-1) o.scale=(o.space=="pos"?.1:.02)/relative_scale;
    if(o.step==-1) o.step=(o.space=="pos"?2:20)*relative_scale;
    for(double x:{o.radius,o.mass,o.eps_factor,o.scale,o.lacunarity,o.axes[0],o.axes[1],o.axes[2]})
        if(!(x>0) || !std::isfinite(x)) throw std::invalid_argument("Scales and dispersions must be finite and positive");
    if(!(o.theta>=0) || !std::isfinite(o.theta) || !(o.veps_factor>=0) || !std::isfinite(o.veps_factor) || !(o.step>=0) || !std::isfinite(o.step) || !(o.persistence>=0) || !std::isfinite(o.persistence))
        throw std::invalid_argument("Invalid tree or Perlin option");
    if((o.diagnostic_counts!=0 && o.diagnostic_counts!=1) ||
        !std::isfinite(o.count_min) || !std::isfinite(o.count_max) ||
        !(o.count_min==-1 || o.count_min>0) || !(o.count_max==-1 || o.count_max>0))
        throw std::invalid_argument("Count bounds require -1 (historical default) or positive finite values; diagnostic-counts is 0 or 1");
    return o;
}
std::ofstream output_file(const fs::path& path) {
    std::ofstream f(path); f.exceptions(std::ios::badbit|std::ios::failbit); f<<std::setprecision(17); return f;
}
double trace(const Diffu_tensor_vel& t) {return t.D[0][0]+t.D[1][1]+t.D[2][2];}
void tensor_row(std::ostream& f,size_t id,const Diffu_tensor_vel& t) {
    f<<id<<'\t'<<trace(t)<<'\t'<<t.D[0][0]<<'\t'<<t.D[1][1]<<'\t'<<t.D[2][2]<<'\t'<<t.D[0][1]<<'\t'<<t.D[0][2]<<'\t'<<t.D[1][2]<<'\n';
}
template<class T> void save_points(const fs::path& path,const std::vector<T>& p) {
    auto f=output_file(path);f<<"# mfrt_trace_v2 id x y z\n";
    for(size_t i=0;i<p.size();++i) f<<i<<'\t'<<p[i].x<<'\t'<<p[i].y<<'\t'<<p[i].z<<'\n';
}
template<class T> std::vector<size_t> select_checks(const std::vector<T>& p,const std::vector<double>& d,size_t n,uint64_t seed) {
    std::vector<size_t> ids(p.size());std::iota(ids.begin(),ids.end(),0);
    if(n>=p.size()) return ids;
    if(!n) return {};
    std::mt19937_64 gen(seed);std::shuffle(ids.begin(),ids.end(),gen);ids.resize(n);
    // Include the strongest diffusion and fastest/outermost targets in addition
    // to the random subset. Their selection is recorded by their source IDs.
    const auto top=std::min(size_t(16),p.size());
    std::vector<size_t> order(p.size());std::iota(order.begin(),order.end(),0);
    std::partial_sort(order.begin(),order.begin()+top,order.end(),[&](size_t a,size_t b){return d[a]>d[b];});
    ids.insert(ids.end(),order.begin(),order.begin()+top);
    auto norm=[&](size_t i){return p[i].x*p[i].x+p[i].y*p[i].y+p[i].z*p[i].z;};
    std::partial_sort(order.begin(),order.begin()+top,order.end(),[&](size_t a,size_t b){return norm(a)>norm(b);});
    ids.insert(ids.end(),order.begin(),order.begin()+top);
    std::sort(ids.begin(),ids.end());ids.erase(std::unique(ids.begin(),ids.end()),ids.end());return ids;
}
int main(int argc,char** argv) {try {
    const Options o=parse(argc,argv);const fs::path out=o.output;
    N_particles=o.n;M_total=o.mass;R0=o.radius;calculate_fixed_const_coefs();
    R_epsilon*=o.eps_factor;v_epsilon*=o.veps_factor;
    const double count_scale=o.space=="pos"?Rm:v0;
    const auto count_bounds=o.count?fractal_count_bounds(o.n,count_scale,o.count_min,o.count_max,o.nr):std::make_pair(0.,0.);
    if(fs::exists(out) && !fs::is_empty(out)) throw std::runtime_error("Output directory is not empty; choose a new run ID");
    fs::create_directories(out);omp_set_dynamic(0);omp_set_num_threads(o.threads);
    set_sample_seed(o.seed);
    const std::vector<double> mean{0,0,0},iso=rescale_dispersion_keep_ratio(mean,{1,1,1},v0),high=rescale_dispersion_keep_ratio(mean,o.axes,v0);
    Eigen::Matrix3d cov_iso=Eigen::Matrix3d::Zero(),cov_high=cov_iso;
    for(int j=0;j<3;++j) {cov_iso(j,j)=iso[j]*iso[j];cov_high(j,j)=high[j]*high[j];}
    std::vector<PositionSpace> pos;std::vector<VelocitySpace> vel;
    double start=omp_get_wtime();
    generate_simple_xv(pos,vel,o.n,Rm,mean,iso);
    auto noise=[&](auto& p){generate_noised_samples(p,o.iterations,o.scale,o.octaves,o.step,o.perlin_seed,o.persistence,o.lacunarity);};
    auto affine=[&](const Eigen::Matrix3d& cov){readjust_velocities_dispersion_affine(vel,Eigen::Vector3d::Zero(),cov);};
    if(o.space=="pos") {
        if(o.kind!="uniform") {noise(pos);readjust_positions(pos,Rm);}
        if(o.kind=="radial") {
            // Preserve each source radius about its original centroid exactly.
            // Do not recenter the randomized sample, which would alter radii.
            std::mt19937_64 gen(o.angular_seed);std::uniform_real_distribution<double> u(0,1);
            for(auto& p:pos) {double r=std::sqrt(p.x*p.x+p.y*p.y+p.z*p.z),z=2*u(gen)-1,phi=2*M_PI*u(gen),s=std::sqrt(1-z*z);p={r*s*std::cos(phi),r*s*std::sin(phi),r*z};}
        }
        save_points(out/"samples.tsv",pos);
    } else {
        if(o.kind=="aniso") affine(cov_high);
        if(o.kind=="tail" || o.kind=="composite") {
            generate_revision_tail_sample(vel,o.n,v0,mean,iso,high,o.kind=="tail");
            save_points(out/"tail_before_affine.tsv",vel);
            affine(cov_iso);
        }
        if(o.kind=="noised" || o.kind=="composite") {noise(vel);affine(o.kind=="composite"?cov_high:cov_iso);}
        if(v_epsilon==0) {
            for(const auto& p:vel)if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z))
                throw std::domain_error("Nonfinite velocity before zero-softening check");
            std::vector<size_t> order(o.n);std::iota(order.begin(),order.end(),0);
            auto key=[&](size_t i){return std::tie(vel[i].x,vel[i].y,vel[i].z);};
            std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return key(a)<key(b);});
            for(size_t i=1;i<order.size();++i)if(key(order[i-1])==key(order[i]))
                throw std::domain_error("Distinct particles have exactly equal velocities: unsoftened kernel undefined. Examine sample and specify a documented positive --veps-factor if required.");
        }
        save_points(out/"samples.tsv",vel);
    }
    const double generate_seconds=omp_get_wtime()-start;
    double build_seconds,tree_seconds,direct_seconds,count_seconds;size_t n_direct;
    std::vector<double> d(o.n);
    if(o.space=="pos") {
        start=omp_get_wtime();Octree tree(o.theta);tree.build(pos);build_seconds=omp_get_wtime()-start;
        start=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,64)
        for(size_t i=0;i<o.n;++i) d[i]=tree.calculateDiffusionCoefficient(pos[i]);
        tree_seconds=omp_get_wtime()-start;
        {auto f=output_file(out/"diffusion.tsv");f<<"# mfrt_trace_v2 id Dtot\n";for(size_t i=0;i<o.n;++i)f<<i<<'\t'<<d[i]<<'\n';}
        auto ids=select_checks(pos,d,o.direct,o.seed^0xd1ec7);n_direct=ids.size();std::vector<double> dd(n_direct);
        start=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,16)
        for(size_t k=0;k<ids.size();++k)dd[k]=directSummation_diffusionCoefficient_pos(pos,pos[ids[k]],ids[k]);
        direct_seconds=omp_get_wtime()-start;
        {auto f=output_file(out/"direct.tsv");f<<"# mfrt_trace_v2 id Dtot\n";for(size_t k=0;k<ids.size();++k)f<<ids[k]<<'\t'<<dd[k]<<'\n';}
        start=omp_get_wtime();if(o.count)calculate_frac_dim(pos,Rm,(out/"nr.tsv").string(),1.,o.seed,o.count,o.count_min,o.count_max,o.nr);count_seconds=omp_get_wtime()-start;
    } else {
        start=omp_get_wtime();Octree_vel tree(o.theta);tree.build(vel);build_seconds=omp_get_wtime()-start;
        std::vector<Diffu_tensor_vel> dt(o.n);start=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,64)
        for(size_t i=0;i<o.n;++i) {tree.calculateDiffusionCoefficient(vel[i],dt[i]);d[i]=trace(dt[i]);}
        tree_seconds=omp_get_wtime()-start;
        {auto f=output_file(out/"diffusion.tsv");f<<"# mfrt_trace_v2 id Dtot D11 D22 D33 D12 D13 D23\n";for(size_t i=0;i<o.n;++i)tensor_row(f,i,dt[i]);}
        auto ids=select_checks(vel,d,o.direct,o.seed^0xd1ec7);n_direct=ids.size();std::vector<Diffu_tensor_vel> dd(n_direct);start=omp_get_wtime();
#pragma omp parallel for schedule(dynamic,16)
        for(size_t k=0;k<ids.size();++k)directSummationVelDiffusion_fraction(vel,vel[ids[k]],ids[k],dd[k]);
        direct_seconds=omp_get_wtime()-start;
        {auto f=output_file(out/"direct.tsv");f<<"# mfrt_trace_v2 id Dtot D11 D22 D33 D12 D13 D23\n";for(size_t k=0;k<ids.size();++k)tensor_row(f,ids[k],dd[k]);}
        start=omp_get_wtime();if(o.count)calculate_frac_dim(vel,v0,(out/"nr.tsv").string(),1.,o.seed,o.count,o.count_min,o.count_max,o.nr);count_seconds=omp_get_wtime()-start;
    }
    if(o.count && o.diagnostic_counts) {
        start=omp_get_wtime();
        if(o.space=="pos")calculate_frac_dim(pos,Rm,(out/"nr_diagnostic.tsv").string(),1.,o.seed,o.count,.001*Rm,Rm,36);
        else calculate_frac_dim(vel,v0,(out/"nr_diagnostic.tsv").string(),1.,o.seed,o.count,.001*v0,v0,36);
        count_seconds+=omp_get_wtime()-start;
    }
    for(double x:d)if(!(x>0)||!std::isfinite(x))throw std::runtime_error("Nonpositive or nonfinite diffusion");
    auto f=output_file(out/"metadata.json");
    f<<"{\n  \"schema\": \""<<diffusion_schema<<"\",\n  \"space\": \""<<o.space<<"\",\n  \"kind\": \""<<o.kind<<"\",\n";
    auto value=[&](const char* key,auto x){f<<"  \""<<key<<"\": "<<x<<",\n";};
    value("n",o.n);value("seed",o.seed);value("angular_seed",o.angular_seed);value("perlin_seed",o.perlin_seed);value("threads",o.threads);
    value("R0",R0);value("Rm",Rm);value("sphere_radius",sphere_radius);value("mass",M_total);value("G",G);value("v0",v0);value("n0",n0);
    value("crossing_length",R0);value("t_cross",R0/v0);value("velocity_second_moment",v0*v0);
    value("epsilon",R_epsilon);value("v_epsilon",v_epsilon);value("eps_factor",o.eps_factor);value("veps_factor",o.veps_factor);value("ln_Lambda",ln_Lambda);
    value("D0",Diffu_0_unit);value("D_classical_median",Diffu_ref);value("D_classical_mean",3*ln_Lambda*Diffu_0_unit*maxwellian_trace_mean_factor);
    value("v50",v0*maxwellian_vmedian_over_vrms);value("GXX",GXX);value("position_sum_prefactor",coef_Diffu_pos_trace);
    value("theta",o.theta);value("iterations",o.iterations);value("noise_scale",o.scale);value("noise_step",o.step);value("octaves",o.octaves);value("persistence",o.persistence);value("lacunarity",o.lacunarity);
    value("noise_scale_times_count_scale",o.scale*count_scale);
    value("noise_step_over_count_scale",o.step/count_scale);
    value("tail_velocity_scale_factor",v0/std::sqrt(G*137./75.));
    f<<"  \"axes\": ["<<o.axes[0]<<','<<o.axes[1]<<','<<o.axes[2]<<"],\n";
    value("count_targets",std::min(o.count,o.n));value("count_min_requested",o.count_min);value("count_max_requested",o.count_max);value("count_radii",o.nr);value("direct_targets",n_direct);value("direct_random_requested",o.direct);
    value("count_scale",count_scale);value("count_r_min",count_bounds.first);value("count_r_max",count_bounds.second);
    value("diagnostic_counts",o.diagnostic_counts);
    value("generate_seconds",generate_seconds);value("build_seconds",build_seconds);value("tree_seconds",tree_seconds);value("direct_seconds",direct_seconds);value("count_seconds",count_seconds);
    f<<"  \"scalar\": \"trace; no factor 1/3 or 1/2\",\n  \"units\": \"length kpc; velocity km/s; mass 1e10 Msun; diffusion (km/s)^3/kpc\",\n  \"position_velocity_convention\": \"fixed Maxwellian median speed v50\",\n  \"radial_origin\": \"original centered perlin sample origin; no recentering after angular randomization\",\n  \"legacy_GXX_correction\": false,\n  \"complete\": true\n}\n";
    std::cout<<"Completed "<<out<<"; tree "<<tree_seconds<<" s, direct "<<direct_seconds<<" s, counts "<<count_seconds<<" s\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<"Diffusion calculation failed: "<<e.what()<<'\n';return 1;}}
