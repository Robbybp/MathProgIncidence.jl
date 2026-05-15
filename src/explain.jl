import JuMP
import MathProgIncidence as MPIN
import LinearAlgebra

include("nlp.jl")

struct ExplainOpts
    point::Function
    dual::Function
    atol::Float64
    max_size::Union{Nothing,Int}
    function ExplainOpts(;
        point::Function = JuMP.value,
        dual::Function = JuMP.dual,
        atol::Float64 = 0.0,
        max_size::Union{Nothing,Int} = nothing,
    )
        return new(point, dual, atol, max_size)
    end
end

# If provided a variable, we "explain" in terms of constraints and the objective.
# If provided a constraint, we just explain in terms of the variables in the constraint.
#
# Would we ever want to provide a "two-step", or nested, explanation?
# - Each explainer, e.g., variable, can come with its own explanation
# - If so, we may want to flatten or aggregated these nested explanations.
# - But probably not by default, as preserving the hierarchical information sounds useful.
#
# Would it be helpful to put the solution into a reduced space?
# - Any constraint/objective explanation is in terms of the degree-of-freedom vars
# - Any variable explanation is in terms of inequalities
# - To keep things orthogonal, it would be nice to just have a ReducedSpaceNLP.
# - The problem is that this doesn't necessarily have all the variables we want
#   to explain.

function explain(nlp::NLP, var::JuMP.VariableRef; opts = ExplainOpts())
    x = map(opts.point, nlp.variables)
    # I really need the Lagrangian gradient as an affine expression
    i = findfirst(v -> v === var,  nlp.variables)
    # I need not just the value of the Lagrangian, I need the components that contribute
    # to it.
    objective = JuMP.objective_function(var.model)
    obj_grad = eval_objective_gradient(nlp, x)
    jacobian = eval_constraint_jacobian(nlp, x)
    igraph = MPIN.IncidenceGraphInterface(var.model; include_inequality = true)
    adjacent_cons = MPIN.get_adjacent(igraph, var)
    con_indices = findall(c -> c in adjacent_cons, nlp.constraints)
    adjacent_cons = nlp.constraints[con_indices]
    λ = opts.dual.(adjacent_cons)
    lagrangian_coefficients = λ .* vec(jacobian[con_indices, i])
    explanation = Dict{Any, Float64}(zip(adjacent_cons, lagrangian_coefficients))
    explanation[objective] = nlp.lagrangian_objective_factor * obj_grad[i]
    return explanation
end

function explain(var::JuMP.VariableRef; opts = ExplainOpts())
    nlp = NLP(var.model)
    return explain(nlp, var; opts)
end

"""
Our optimization problem is:

    min  f(x, y)
    s.t. g(x, y) = 0
         h(x, y) ≤ 0

We will eliminate g and y.
"""
function explain(
    nlp::NLP,
    var::JuMP.VariableRef,
    eliminated_vars::Vector{JuMP.VariableRef},
    eliminated_cons::Vector{JuMP.ConstraintRef};
    opts = ExplainOpts(),
)
    all_var_values = map(opts.point, nlp.variables)
    y_indices = findall(v -> v ∈ eliminated_vars, nlp.variables) # TODO: Fix this quadratic loop
    x_indices = findall(v -> v ∉ eliminated_vars, nlp.variables)
    #y = all_var_values[y_indices]
    #x = all_var_values[x_indices]
    g_indices = findall(c -> c ∈ eliminated_cons, nlp.constraints)
    h_indices = findall(c -> c ∉ eliminated_cons, nlp.constraints)
    objective = JuMP.objective_function(var.model)
    obj_grad = eval_objective_gradient(nlp, all_var_values)
    jacobian = eval_constraint_jacobian(nlp, all_var_values)

    dfdx = reshape(obj_grad[x_indices], 1, :)
    dfdy = reshape(obj_grad[y_indices], 1, :)
    dgdx = jacobian[g_indices, x_indices]
    dgdy = jacobian[g_indices, y_indices]
    dhdx = jacobian[h_indices, x_indices]
    dhdy = jacobian[h_indices, y_indices]

    dgdy_lu = LinearAlgebra.lu(dgdy)
    dydx = LinearAlgebra.ldiv(dgdy_lu,  dgdx)
    df_reduced_dx = dfdx - dfdy * dydx
    dh_reduced_dx = dhdx - dhdy * dydx

    if var in eliminated_vars
        idx = findfirst(v -> v === var, nlp.variables[y_indices])
        index_coefs = [i => val for (i, val) in enumerate(vec(dydx[idx, :]))]
    else
        idx = findfirst(v -> v === var, nlp.variables[x_indices])
        index_coefs = [idx => 1.0]
    end

    retained_constraints = nlp.constraints[h_indices]
    λh = JuMP.dual.(retained_constraints)
    explanation = Dict{Any,Float64}(zip(retained_constraints, zeros(length(retained_constraints))))
    explanation[objective] = 0.0
    for (i, coef) in index_coefs
        # Compute explanation of the indicated variables
        lagrangian_coefficients = λh .* vec(dh_reduced_dx[:, i])
        var_explanation = Dict{Any,Float64}(zip(retained_constraints, lagrangian_coefficients))
        var_explanation[objective] = nlp.lagrangian_objective_factor * df_reduced_dx[1, i]
        # Weigh the explanation for each independent variable by our target variable's
        # derivative with respect to it.
        for (k, val) in var_explanation
            explanation[k] += coef * val
        end
    end
    explanation = filter(e -> abs(e.second) >= opts.atol, explanation)
    return explanation
end

function explain(
    var::JuMP.VariableRef,
    eliminated_vars::Vector{JuMP.VariableRef},
    eliminated_cons::Vector{JuMP.ConstraintRef};
    opts = ExplainOpts(),
)
    nlp = NLP(var.model)
    return explain(nlp, var, eliminated_vars, eliminated_cons; opts)
end
