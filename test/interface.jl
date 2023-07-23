#  ___________________________________________________________________________
#
#  JuMPIn.jl: JuMP Incidence Graph Analysis
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

using Test: @test, @test_throws, @testset
import JuMP
import JuMPIn as ji

include("models.jl") # make_degenerate_flow_model

function _test_igraph_fields(igraph, constraints, variables)
    @test Set(variables) == keys(igraph._var_node_map)
    @test Set(constraints) == keys(igraph._con_node_map)

    ncon = length(constraints)
    nvar = length(variables)

    var_nodes = Set(values(igraph._var_node_map))
    con_nodes = Set(values(igraph._con_node_map))
    @test con_nodes == Set(1:ncon)
    @test var_nodes == Set(ncon+1:ncon+nvar)

    @test Set(igraph._nodes[1:ncon]) == Set(constraints)
    @test Set(igraph._nodes[ncon+1:ncon+nvar]) == Set(variables)
end


function test_construct_interface()
    m = make_degenerate_flow_model()
    igraph = ji.IncidenceGraphInterface(m)

    variables = [
        m[:x][1],
        m[:x][2],
        m[:x][3],
        m[:flow_comp][1],
        m[:flow_comp][2],
        m[:flow_comp][3],
        m[:flow],
        m[:rho],
    ]
    constraints = [
        m[:sum_comp_eqn],
        m[:comp_dens_eqn][1],
        m[:comp_dens_eqn][2],
        m[:comp_dens_eqn][3],
        m[:bulk_dens_eqn],
        m[:comp_flow_eqn][1],
        m[:comp_flow_eqn][2],
        m[:comp_flow_eqn][3],
    ]
    _test_igraph_fields(igraph, constraints, variables)
    return nothing
end


function test_construct_interface_rectangular()
    m = make_degenerate_flow_model()
    @JuMP.constraint(
        m,
        sum_flow_eqn,
        m[:flow] == sum(m[:flow_comp][:]),
    )
    igraph = ji.IncidenceGraphInterface(m)

    variables = [
        m[:x][1],
        m[:x][2],
        m[:x][3],
        m[:flow_comp][1],
        m[:flow_comp][2],
        m[:flow_comp][3],
        m[:flow],
        m[:rho],
    ]
    constraints = [
        m[:sum_comp_eqn],
        m[:comp_dens_eqn][1],
        m[:comp_dens_eqn][2],
        m[:comp_dens_eqn][3],
        m[:bulk_dens_eqn],
        m[:comp_flow_eqn][1],
        m[:comp_flow_eqn][2],
        m[:comp_flow_eqn][3],
        m[:sum_flow_eqn],
    ]
    _test_igraph_fields(igraph, constraints, variables)
    return nothing
end


function test_get_adjacent_to_linear_constraint()
    m = make_degenerate_flow_model()
    igraph = ji.IncidenceGraphInterface(m)
    con = m[:sum_comp_eqn]
    adjacent = ji.get_adjacent(igraph, con)
    @test Set(adjacent) == Set([m[:x][1], m[:x][2], m[:x][3]])
    return nothing
end


function test_get_adjacent_to_quadratic_constraint()
    m = make_degenerate_flow_model()
    igraph = ji.IncidenceGraphInterface(m)
    con = m[:comp_dens_eqn][1]
    adjacent = ji.get_adjacent(igraph, con)
    @test Set(adjacent) == Set([m[:x][1], m[:rho]])
    return nothing
end


function test_get_adjacent_to_nonlinear_constraint()
    m = make_degenerate_flow_model()
    igraph = ji.IncidenceGraphInterface(m)
    con = m[:bulk_dens_eqn]
    adjacent = ji.get_adjacent(igraph, con)
    @test Set(adjacent) == Set([m[:x][1], m[:x][2], m[:x][3], m[:rho]])
    return nothing
end


function test_get_adjacent_to_variable()
    m = make_degenerate_flow_model()
    igraph = ji.IncidenceGraphInterface(m)
    var = m[:x][2]
    adjacent = ji.get_adjacent(igraph, var)
    incident_cons = [
        m[:sum_comp_eqn],
        m[:bulk_dens_eqn],
        m[:comp_dens_eqn][2],
        m[:comp_flow_eqn][2],
    ]
    @test Set(adjacent) == Set(incident_cons)
    return nothing
end


function test_maximum_matching()
    m = make_degenerate_flow_model()
    igraph = ji.IncidenceGraphInterface(m)
    matching = ji.maximum_matching(igraph)
    @test length(matching) == 7
    for (con, var) in matching
        @test typeof(con) <: JuMP.ConstraintRef
        @test typeof(var) <: JuMP.VariableRef
        @test var in Set(ji.get_adjacent(igraph, con))
        @test con in Set(ji.get_adjacent(igraph, var))
    end
    possibly_unmatched_vars = Set([
        m[:flow_comp][1],
        m[:flow_comp][2],
        m[:flow_comp][3],
        m[:flow],
    ])
    possibly_unmatched_cons = Set([
        m[:comp_dens_eqn][1],
        m[:comp_dens_eqn][2],
        m[:comp_dens_eqn][3],
        m[:bulk_dens_eqn],
        m[:sum_comp_eqn],
    ])
    for con in keys(igraph._con_node_map)
        if !(con in keys(matching))
            @test con in possibly_unmatched_cons
        end
    end
    matched_var_set = Set(values(matching))
    for var in keys(igraph._var_node_map)
        if !(var in matched_var_set)
            @test var in possibly_unmatched_vars
        end
    end
    return nothing
end


function test_dulmage_mendelsohn()
    m = make_degenerate_flow_model()
    igraph = ji.IncidenceGraphInterface(m)
    con_dmp, var_dmp = ji.dulmage_mendelsohn(igraph)
    con_undercon = con_dmp.underconstrained
    con_overcon = cat(con_dmp.overconstrained, con_dmp.unmatched, dims=1)
    @test Set(con_undercon) == Set([
        m[:comp_flow_eqn][1], m[:comp_flow_eqn][2], m[:comp_flow_eqn][3]
    ])
    @test Set(con_overcon) == Set([
        m[:comp_dens_eqn][1],
        m[:comp_dens_eqn][2],
        m[:comp_dens_eqn][3],
        m[:bulk_dens_eqn],
        m[:sum_comp_eqn],
    ])
    @test con_dmp.square == []
    var_undercon = cat(var_dmp.underconstrained, var_dmp.unmatched, dims=1)
    var_overcon = var_dmp.overconstrained
    @test Set(var_undercon) == Set([
        m[:flow_comp][1], m[:flow_comp][2], m[:flow_comp][3], m[:flow]
    ])
    @test Set(var_overcon) == Set([m[:x][1], m[:x][2], m[:x][3], m[:rho]])
    @test var_dmp.square == []
    return nothing
end

function test_overconstrained_due_to_fixed_variable()
    m = JuMP.Model()
    @JuMP.variable(m, x[1:2])
    @JuMP.constraint(m, x[1] + 2*x[2] == 1)
    @JuMP.constraint(m, 3*x[2] - x[2] == 0)
    JuMP.fix(x[1], 3)
    igraph = ji.IncidenceGraphInterface(m)
    con_dmp, var_dmp = ji.dulmage_mendelsohn(igraph)
    @test length(var_dmp.overconstrained) == 2
    @test length(con_dmp.overconstrained) == 2
    @test length(con_dmp.unmatched) == 1
    return
end

function test_overconstrained_due_to_including_bound()
    m = JuMP.Model()
    @JuMP.variable(m, x)
    @JuMP.variable(m, 0.01 <= y)
    @JuMP.constraint(m, 2*x + y == 1)
    @JuMP.NLconstraint(m, x == sqrt(y))
    igraph = ji.IncidenceGraphInterface(m, include_inequality = true)
    con_dmp, var_dmp = ji.dulmage_mendelsohn(igraph)
    @test length(var_dmp.overconstrained) == 2
    @test length(con_dmp.overconstrained) == 2
    @test length(con_dmp.unmatched) == 1
    return
end

function test_interface_from_constraints_and_variables()
    m = JuMP.Model()
    @JuMP.variable(m, x[1:3])
    @JuMP.constraint(m, eq1,  x[1] + x[2] == 2)
    @JuMP.constraint(m, eq2, x[3]*x[2] == 1.1)
    constraints = [eq1, eq2]
    variables = [x[1], x[3]]
    igraph = ji.IncidenceGraphInterface(constraints, variables)
    _test_igraph_fields(igraph, constraints, variables)
    return
end

function test_matching_from_constraints_and_variables()
    m = JuMP.Model()
    @JuMP.variable(m, x[1:3])
    @JuMP.constraint(m, eq1,  x[1] + x[2] == 2)
    @JuMP.constraint(m, eq2, x[3]*x[2] == 1.1)
    constraints = [eq1, eq2]
    variables = [x[1], x[3]]
    matching = ji.maximum_matching(constraints, variables)
    @test length(matching) == 2
    @test matching[eq1] == x[1]
    @test matching[eq2] == x[3]
    return
end

function test_dulmage_mendelsohn_from_constraints_and_variables()
    m = JuMP.Model()
    @JuMP.variable(m, x[1:3])
    @JuMP.constraint(m, eq1,  x[1] + x[2] == 2)
    @JuMP.constraint(m, eq2, x[3]*x[2] == 1.1)
    constraints = [eq1, eq2]
    variables = [x[1], x[3]]
    con_dmp, var_dmp = ji.dulmage_mendelsohn(constraints, variables)
    @test con_dmp.unmatched == []
    @test con_dmp.underconstrained == []
    @test con_dmp.overconstrained == []
    @test Set(con_dmp.square) == Set(constraints)
    @test var_dmp.unmatched == []
    @test var_dmp.underconstrained == []
    @test var_dmp.overconstrained == []
    @test Set(var_dmp.square) == Set(variables)
    return
end

@testset "interface" begin
    test_construct_interface()
    test_construct_interface_rectangular()
    test_get_adjacent_to_linear_constraint()
    test_get_adjacent_to_quadratic_constraint()
    test_get_adjacent_to_nonlinear_constraint()
    test_get_adjacent_to_variable()
    test_maximum_matching()
    test_dulmage_mendelsohn()
    test_overconstrained_due_to_fixed_variable()
    test_overconstrained_due_to_including_bound()
    test_interface_from_constraints_and_variables()
    test_matching_from_constraints_and_variables()
    test_dulmage_mendelsohn_from_constraints_and_variables()
end
