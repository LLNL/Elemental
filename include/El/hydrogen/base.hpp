////////////////////////////////////////////////////////////////////////////////xecu
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
//
// lbann_base .hpp - Basic definitions, functions
////////////////////////////////////////////////////////////////////////////////

#ifndef BASE_HPP
#define BASE_HPP

#include "El.hpp"

// Typedefs for Elemental matrices
typedef float DataType;

using EGrid      = El::Grid;
using Grid       = El::Grid;
using Mat        = El::Matrix<DataType>;
using AbsDistMat = El::AbstractDistMatrix<DataType>;
using ElMat      = El::ElementalMatrix<DataType>;
using BlockMat   = El::BlockMatrix<DataType>;
using MCMRMat    = El::DistMatrix<DataType, El::MC  , El::MR  >;
using CircMat    = El::DistMatrix<DataType, El::CIRC, El::CIRC>;
using StarMat    = El::DistMatrix<DataType, El::STAR, El::STAR>;
using StarVCMat  = El::DistMatrix<DataType, El::STAR, El::VC  >;
using VCStarMat  = El::DistMatrix<DataType, El::VC  , El::STAR>;
using MCStarMat  = El::DistMatrix<DataType, El::MC  , El::STAR>;
using MRStarMat  = El::DistMatrix<DataType, El::MR  , El::STAR>;
using StarMRMat  = El::DistMatrix<DataType, El::STAR, El::MR  >;

// Deprecated typedefs for Elemental matrices
using DistMat         = MCMRMat;
using RowSumMat       = MCStarMat;
using ColSumStarVCMat = VCStarMat;
using ColSumMat       = MRStarMat;

// Datatype for model evaluation
// Examples: timing, metrics, objective functions
using EvalType = double;

/// Distributed matrix format
enum class matrix_format {MC_MR, CIRC_CIRC, STAR_STAR, STAR_VC, MC_STAR, invalid};

/// Data layout that is optimized for different modes of parallelism
enum class data_layout {MODEL_PARALLEL, DATA_PARALLEL, invalid};
static matrix_format __attribute__((used)) data_layout_to_matrix_format(data_layout layout) {
  matrix_format format;
  switch(layout) {
  case data_layout::MODEL_PARALLEL:
    format = matrix_format::MC_MR;
    break;
  case data_layout::DATA_PARALLEL:
    /// Weights are stored in STAR_STAR and data in STAR_VC
    format = matrix_format::STAR_STAR;
    break;
  default:
    throw(std::string{} + __FILE__ + " " + std::to_string(__LINE__) + " Invalid data layout selected");
  }
  return format;
}

/// Neural network execution mode
enum class execution_mode {training, validation, testing, prediction, invalid};
static const char *__attribute__((used)) _to_string(execution_mode m) {
  switch(m) {
  case execution_mode::training:
    return "training";
  case execution_mode::validation:
    return "validation";
  case execution_mode::testing:
    return "testing";
  case execution_mode::prediction:
    return "prediction";
  case execution_mode::invalid:
    return "invalid";
  default:
    throw("Invalid execution mode specified"); /// @todo this should be an lbann_exception but then the class has to move to resolve dependencies
  }
}

/** Pooling layer mode */
enum class pool_mode {invalid, max, average, average_no_pad};

/** returns a string representation of the pool_mode */
std::string get_pool_mode_name(pool_mode m);


#endif // BASE_HPP
