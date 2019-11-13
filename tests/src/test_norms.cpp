#include "catch2/catch.hpp"

#include <cstddef>
#include <cmath>

#include <algorithm>
#include <vector>

#include "moab/Core.hpp"

#include "blas.hpp"

#include "MeshLevel.hpp"
#include "UniformMeshHierarchy.hpp"
#include "norms.hpp"

#include "testing_utilities.hpp"

static const double inf = std::numeric_limits<double>::infinity();

//We could test the `L^inf` and `L^2` norms on a tetrahedral mesh, but I'll wait
//until we have 3D refinement done and can test the `s` norms as well.
TEST_CASE("basic norm properties", "[norms]") {
    moab::ErrorCode ecode;
    moab::Core mbcore;
    ecode = mbcore.load_file(mesh_path("pyramid.msh").c_str());
    require_moab_success(ecode);
    mgard::MeshLevel mesh(mbcore);
    mgard::UniformMeshHierarchy hierarchy(mesh, 4);
    const std::size_t N = hierarchy.ndof();
    std::vector<double> smoothness_parameters = {
        -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, inf
    };

    std::random_device device;
    std::default_random_engine generator(device());
    std::uniform_real_distribution<double> distribution(-1, 1);

    SECTION("absolute homogeneity") {
        std::vector<double> u_(N);
        double * const u = u_.data();
        //Likely not needed. Not checking now.
        std::fill(u_.begin(), u_.end(), 0);
        for (const double s : smoothness_parameters) {
            REQUIRE(mgard::norm(u, hierarchy, s) == 0);
        }

        for (double &value : u_) {
            value = distribution(generator);
        }

        std::vector<double> copy_(N);
        double * const copy = copy_.data();
        for (const double s : smoothness_parameters) {
            blas::copy(N, u, copy);
            const double alpha = distribution(generator);
            blas::scal(N, alpha, copy);
            REQUIRE(mgard::norm(copy, hierarchy, s) == Approx(
                std::abs(alpha) * mgard::norm(u, hierarchy, s)
            ));
        }
    }

    SECTION("triangle inequality") {
        std::vector<double> u_(N);
        double * const u = u_.data();
        std::vector<double> v_(N);
        double * const v = v_.data();
        for (double &value : u_) {
            value = distribution(generator);
        }
        for (double &value : v_) {
            value = distribution(generator);
        }
        std::vector<double> w_ = u_;
        double * const w = w_.data();
        blas::axpy(N, 1.0, v, u);
        for (const double s : smoothness_parameters) {
            REQUIRE(
                mgard::norm(w, hierarchy, s) <=
                mgard::norm(u, hierarchy, s) + mgard::norm(v, hierarchy, s)
            );
        }
    }
}

TEST_CASE("comparison with Python implementation: norms", "[norms]") {
    moab::ErrorCode ecode;
    moab::Core mbcore;
    ecode = mbcore.load_file(mesh_path("circle.msh").c_str());
    require_moab_success(ecode);
    mgard::MeshLevel mesh(mbcore);
    mgard::UniformMeshHierarchy hierarchy(mesh, 2);
    const std::size_t N = hierarchy.ndof();

    std::vector<double> u_(N);
    double * const u = u_.data();

    const moab::Range &NODES = hierarchy.meshes.back().entities[moab::MBVERTEX];
    for (std::size_t i = 0; i < N; ++i) {
        double xyz[3];
        const moab::EntityHandle node = NODES[i];
        mbcore.get_coords(&node, 1, xyz);
        const double x = xyz[0];
        const double y = xyz[1];
        const double z = xyz[2];
        u_.at(i) = std::sin(5 * x) + 2 * std::cos(32 * y) - std::sin(x * y * z);
    }

    REQUIRE(mgard::norm(u, hierarchy,  inf) == Approx(2.99974381309398));
    REQUIRE(mgard::norm(u, hierarchy, -1.5) == Approx(1.041534180771523));
    REQUIRE(mgard::norm(u, hierarchy, -1.0) == Approx(1.086120609647959));
    REQUIRE(mgard::norm(u, hierarchy, -0.5) == Approx(1.1720508380006622));
    REQUIRE(mgard::norm(u, hierarchy,  0.0) == Approx(1.3338133542219779));
    REQUIRE(mgard::norm(u, hierarchy, 1e-9) == Approx(1.3338133546552449));
    REQUIRE(mgard::norm(u, hierarchy,  0.5) == Approx(1.6305906723975383));
    REQUIRE(mgard::norm(u, hierarchy,  1.0) == Approx(2.1667011853555294));
    REQUIRE(mgard::norm(u, hierarchy,  1.5) == Approx(3.14182423518829));
}
