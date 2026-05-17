import JuMP
import MathOptInterface as MOI
import SparseArrays

module _NLP

struct NLP
    variables::Vector{JuMP.VariableRef}
    constraints::Vector{<:JuMP.ConstraintRef}
    constraint_ubs::Vector{Float64}
    constraint_lbs::Vector{Float64}
    eq_indices::Vector{Int}
    ineq_indices::Vector{Int}
    variable_to_index::Dict{JuMP.VariableRef,Int}
    constraint_to_index::Dict{<:JuMP.ConstraintRef,Int}
    lagrangian_objective_factor::Int
    evaluator::MOI.AbstractNLPEvaluator
    function NLP(model::JuMP.Model)
        nlp = MOI.Nonlinear.Model()
        variables = JuMP.all_variables(model)
        constraints = JuMP.all_constraints(model; include_variable_in_set_constraints = true)
        ncon = length(constraints)
        constraint_ubs = fill(Inf, ncon)
        constraint_lbs = fill(-Inf, ncon)
        eq_indices = Int[]
        ineq_indices = Int[]
        for (i, cref) in enumerate(constraints)
            # What to do about vector functions?
            @assert cref.shape == JuMP.ScalarShape()
            con = JuMP.constraint_object(cref)
            MOI.Nonlinear.add_constraint(nlp, con.func, con.set)
            if con.set isa MOI.EqualTo
                push!(eq_indices, i)
                constraint_ubs[i] = con.set.value
                constraint_lbs[i] = con.set.value
            elseif con.set isa MOI.LessThan
                push!(ineq_indices, i)
                constraint_ubs[i] = con.set.upper
            elseif con.set isa MOI.GreaterThan
                push!(ineq_indices, i)
                constraint_lbs[i] = con.set.lower
            elseif con.set isa MOI.Interval
                push!(ineq_indices, i)
                constraint_lbs[i] = con.set.lower
                constraint_ubs[i] = con.set.upper
            else
                error("Unsupported constraint set $(con.set)")
            end
        end
        variable_to_index = Dict(v => i for (i, v) in enumerate(variables))
        constraint_to_index = Dict(c => i for (i, c) in enumerate(constraints))
        MOI.Nonlinear.set_objective(nlp, JuMP.objective_function(model))
        # The purpose of the objective factor is to make sure that the objective-gradient
        # term in the gradient of the Lagrangian is a direction of improvement.
        # Note that we solve *minimization and maximization* problems, and this factor
        # is only for the purpose of computing the Lagrangian.
        lagrangian_objective_factor = JuMP.objective_sense(model) == JuMP.MIN_SENSE ? -1 : 1
        evaluator = MOI.Nonlinear.Evaluator(nlp, MOI.Nonlinear.SparseReverseMode(), JuMP.index.(variables))
        MOI.initialize(evaluator, [:Grad, :Jac, :Hess])
        return new(
            variables,
            constraints,
            constraint_ubs,
            constraint_lbs,
            eq_indices,
            ineq_indices,
            variable_to_index,
            constraint_to_index,
            lagrangian_objective_factor,
            evaluator,
        )
   end
end

function indices(nlp, variables::Vector{JuMP.VariableRef})
    return map(x -> nlp.variable_to_index[x], variables)
end

function indices(nlp, constraints::Vector{<:JuMP.ConstraintRef})
    return map(c -> nlp.constraint_to_index[c], constraints)
end

function eval_objective(nlp, x)
    return MOI.eval_objective(nlp.evaluator, x)
end

function eval_objective_gradient(nlp, x)
    grad = zeros(length(nlp.variables))
    MOI.eval_objective_gradient(nlp.evaluator, grad, x)
    return grad
end

function eval_constraints(nlp, x)
    g = zeros(length(nlp.constraints))
    MOI.eval_constraint(nlp.evaluator, g, x)
    return g
end

function eval_constraint_jacobian(nlp, x)
    structure = MOI.jacobian_structure(nlp.evaluator)
    values = zeros(length(structure))
    MOI.eval_constraint_jacobian(nlp.evaluator, values, x)
    rows = first.(structure)
    cols = last.(structure)
    m = length(nlp.constraints)
    n = length(nlp.variables)
    jac = SparseArrays.sparse(rows, cols, values, m, n)
    return jac
end

function eval_lagrangian_gradient(nlp, x, λ)
    grad_obj = eval_objective_gradient(nlp, x)
    jac = eval_constraint_jacobian(nlp, x)
    grad_lagrangian = grad_obj * nlp.lagrangian_objective_factor + jac' * λ
    #grad_lagrangian = (
    #    grad_obj * nlp.obj_factor # objective factor makes this a direction of improvement
    #    + jac.eq' * λ.eq # Sign is arbitrary for an EQ constraint
    #    + jac.lt' * λ.lt # Dual is negative for a LT constraint. Interior direction.
    #    + jac.gt' * λ.gt # Dual is positive for a GT constraint. Interior direction.
    #    + jac.interval' * λ.interval # Sign of dual depends on which side is active
    #)
    return grad_lagrangian
end

function eval_lagrangian_hessian(nlp, x, λ)
    structure = MOI.hessian_lagrangian_structure(nlp.evaluator)
    values = zeros(length(structure))
    MOI.eval_hessian_lagrangian(
        nlp.evaluator,
        values,
        x,
        Float64(nlp.lagrangian_objective_factor),
        λ,
    )
    rows = first.(structure)
    cols = last.(structure)
    n = length(nlp.variables)
    hessian = SparseArrays.sparse(rows, cols, values, n, n)
    # TODO: Should this return the full Hessian or just a triangle?
    hessian = (hessian + hessian' - LinearAlgebra.Diagonal(hessian))
    return hessian
end

end
