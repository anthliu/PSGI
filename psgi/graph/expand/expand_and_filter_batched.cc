#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <map>

namespace py = pybind11;

using array_int8 = py::array_t<int8_t>;

uint64_t _encode_binary(const int8_t* arr, long int length) {
    uint64_t ret = 0;
    for (int j = 0; j < (int) length; ++ j) {
        int8_t xj = arr[j];
        if (xj != 0 && xj != 1) {  // must be binary
            throw std::invalid_argument("The input value must be binary.");
        }
        ret += (1ULL << j) * xj;
    }
    return ret;
}

py::tuple expand_and_filter(
        array_int8 expandable_comp, array_int8 expandable_elig,
        array_int8 constraint_comp, array_int8 constraint_elig
        ) {
    // 0. Sanity check.
    py::buffer_info bi_e_comp = expandable_comp.request();
    py::buffer_info bi_e_elig = expandable_elig.request();
    py::buffer_info bi_c_comp = constraint_comp.request();
    py::buffer_info bi_c_elig = constraint_elig.request();
    if (bi_e_comp.ndim != 2) throw std::invalid_argument("expandable_comp must be 2-dimensional.");
    if (bi_e_elig.ndim != 2) throw std::invalid_argument("expandable_elig must be 2-dimensional.");
    if (bi_c_comp.ndim != 2) throw std::invalid_argument("constraint_comp must be 2-dimensional.");
    if (bi_c_elig.ndim != 2) throw std::invalid_argument("constraint_elig must be 2-dimensional.");

    long int V = bi_e_comp.shape[1];
    if (V != bi_e_elig.shape[1] || V != bi_c_comp.shape[1] || V != bi_c_elig.shape[1]) {
        throw std::invalid_argument("All inputs must have same number of subtasks.");
    }
    if (V > 60) {
        throw std::invalid_argument("the number of subtasks cannot be greater than 60.");
    }
    long int E = bi_e_comp.shape[0];
    if (E != bi_e_elig.shape[0]) {
        throw std::invalid_argument("expandable_comp and expandable_elig must have same number of rows.");
    }
    long int C = bi_c_comp.shape[0];
    if (C != bi_c_elig.shape[0]) {
        throw std::invalid_argument("constraint_comp and constraint_elig must have same number of rows.");
    }

    // 1. Build a hash table for constraint. There should be no DC(-1) here.
    const int8_t* ccomp = (const int8_t*) bi_c_comp.ptr;
    const int8_t* celig = (const int8_t*) bi_c_elig.ptr;
    std::unordered_map<uint64_t, uint64_t> constraint_map;

    for (int i = 0; i < C; ++ i) {
        const uint64_t comp_i_encoded = _encode_binary(&ccomp[i * V], V);
        const uint64_t elig_i_encoded = _encode_binary(&celig[i * V], V);

        auto it = constraint_map.find(comp_i_encoded);
        if (it != constraint_map.end() && it->second != elig_i_encoded) {
            throw std::runtime_error("Inconsistent data found on the constraint.");
        }
        constraint_map[comp_i_encoded] = elig_i_encoded;
        // std::cerr << "[DEBUG] " << comp_i_encoded << " -> " << elig_i_encoded << std::endl;
    }

    // 2. Enumerate all possible configurations by expanding the expandable.
    const int8_t* ecomp = (const int8_t*) bi_e_comp.ptr;
    const int8_t* eelig = (const int8_t*) bi_e_elig.ptr;
    std::vector<std::vector<int8_t>> expanded_comp, expanded_elig;

    for (int i = 0; i < E; ++ i) {
        const int8_t* comp_i = &ecomp[i * V];   // array(V)
        const int8_t* elig_i = &eelig[i * V];   // array(V)

        auto is_compatible = [&](const std::vector<int8_t> comp_i_expanded) {
            // lookup the constraint table.
            const uint64_t comp_i_encoded = _encode_binary(&comp_i_expanded[0], V);
            auto it = constraint_map.find(comp_i_encoded);
            if (it == constraint_map.end()) {
                return true;  // no constraint found --> it's compatible.
            }

            const uint64_t elig_constraint_encoded = it->second;
            for (int j = 0; j < V; ++ j) if (elig_i[j] != -1) {
                int constraint_j = !!((1ULL << j) & elig_constraint_encoded);
                if (constraint_j != elig_i[j]) {
                    return false;
                }
            }
            return true;
        };

        // enumerate all possibilties over missing values.
        std::vector<int8_t> comp_i_expanded(V);
        int number_of_dontcares = 0;
        for (int j = 0; j < V; ++ j) number_of_dontcares += (int)(comp_i[j] == -1);
        for (uint64_t s = 0; s < (1ULL << number_of_dontcares); ++ s) {
            for (int j = 0, c = 0; j < V; ++ j) {
                comp_i_expanded[j] = (comp_i[j] != -1) ? comp_i[j] : !!((1ULL << c++) & s);
            }
            // Handle the candidate. If constraint is satisfied, extend the result.
            if (is_compatible(comp_i_expanded)) {
                expanded_comp.push_back(comp_i_expanded);
                expanded_elig.emplace_back(elig_i, elig_i + V);
                // std::cerr << "[DEBUG] i = " << i << ", s = " << s << ", Compatible" << std::endl;
            } else {
                // std::cerr << "[DEBUG] i = " << i << ", s = " << s << ", NOT Compatible" << std::endl;
            }
        }
    }

    // 3. Done. convert the vectors into numpy arrays.
    long int R = (int) expanded_comp.size();
    auto result_comp = array_int8(std::vector<int> {(int) R, (int) V});
    auto result_elig = array_int8(std::vector<int> {(int) R, (int) V});
    int8_t* rcomp = (int8_t*) result_comp.request().ptr;
    int8_t* relig = (int8_t*) result_elig.request().ptr;
    for (int i = 0; i < R; ++ i) {
        for (int j = 0; j < V; ++ j) {
            rcomp[i * V + j] = expanded_comp[i][j];
            relig[i * V + j] = expanded_elig[i][j];
        }
    }
    return py::make_tuple(result_comp, result_elig);
}

PYBIND11_MODULE(expand_and_filter, m) {
    m.doc() = "The expand_and_filter module for Bayesian ILP.";
    m.def("expand_and_filter", &expand_and_filter, "Expand completion and eligibility under a constraint.");
}
