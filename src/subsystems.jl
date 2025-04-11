#  ___________________________________________________________________________
#
#  MathProgIncidence.jl: Math Programming Incidence Graph Analysis
#  Copyright (c) 2023. Triad National Security, LLC. All rights reserved.
#
#  This program was produced under U.S. Government contract 89233218CNA000001
#  for Los Alamos National Laboratory (LANL), which is operated by Triad
#  National Security, LLC for the U.S. Department of Energy/National Nuclear
#  Security Administration. All rights in the program are reserved by Triad
#  National Security, LLC, and the U.S. Department of Energy/National Nuclear
#  Security Administration. The Government is granted for itself and others
#  acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
#  in this material to reproduce, prepare derivative works, distribute copies
#  to the public, perform publicly and display publicly, and to permit others
#  to do so.
#
#  This software is distributed under the 3-clause BSD license.
#  ___________________________________________________________________________

"""Utilities for creating and solving subsystems.
"""

import JuMP

"""
    create_subsystems

Return the JuMP models containing the specified constraints and variables.

TODO: What do to about

"""
function create_subsystems(
    model::JuMP.Model,
    subsystems::Vector{Tuple{Vector, Vector}},
)
    models = []
    return models
end

"""
- each model must be loaded with its own optimizer?
- Then what is the benefit of having this function over just optimizing
  in a loop.
- We also transfer variable values from previous solves into the subsequent
  models.
- These aren't just a sequence of independent models, they are linked by their
  variables.
  - They have the same variables?
  - Or they have some linking among their variables?
"""
function optimize_subsystems!(
    models::Vector{JuMP.model},
)
    @assert !isempty(models)
    variables = JuMP.all_variables(models[1])

    

    for (i, model) in enumerate(models)
        JuMP.optimize!(model)
    end
end
