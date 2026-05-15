# I need:
# - Constraints and variables that define the implicit function
# - The ability to evaluate the implicit function and compute its derivatives
# - Other constraints that use the implicit constraint's output

import NaermLib, JuMP, GasModels, Ipopt, MathProgIncidence as MPIN
import MathOptInterface as MOI
import LinearAlgebra
import Random
import SparseArrays

include("slack.jl")
include("dof.jl")
include("square.jl")
include("nlp.jl")

function _correct_data!(data::Dict)
    correct_slack_nodes!(data)
    if haskey(data, "compressor")
        for (_, c) in data["compressor"]
            c["directionality"] = 1
        end
    end
    return data
end

function implicit_model(data::Dict)
    gm = GasModels.instantiate_model(data, GasModels.WPGasModel, GasModels.build_ogf)
    # x should be the set of all variables that share constraints with y.
    # This **just happens** to be the DOF.
    dof = degrees_of_freedom(gm)
    # We need dependent variables to be ordered consistently across different models.
    # This is not the case is we use all_variables on the square model.
    dependent = get_dependent_system(gm)
    make_model_square!(gm)
    variables = JuMP.all_variables(gm.model)
    dofset = Set(dof)
    dependent_variables = filter(∉(dofset), variables)
    @assert Set(dependent_variables) == Set(dependent.variables)
    # These constraints change every time we re-fix the DOF, but I think this is okay...
    # This is okay because we never re-construct the NLP. These equations, and perhaps
    # their indices in the constraint order, will get out-of-sync, but the indices in
    # the original NLP of "implicit function equations", g(x,y)=0, never change.
    dof_equations = JuMP.FixRef.(dof)
    dof_eqset = Set(dof_equations)
    equations = filter(∉(dof_eqset), JuMP.all_constraints(gm.model; include_variable_in_set_constraints = true))
    @assert Set(equations) == Set(dependent.equations)
    nlp = NLP(gm.model)
    return (;
        model = gm.model,
        x = dof,
        y = dependent.variables,
        equations = dependent.equations,
        nlp,
        x_indices = indices(nlp, dof),
        y_indices = indices(nlp, dependent.variables),
        # I need to know these indices because I want to ignore the x-fixing equations
        eq_indices = indices(nlp, dependent.equations),
    )
end

function residual_model(data::Dict)
    gm = GasModels.instantiate_model(data, GasModels.WPGasModel, GasModels.build_ogf)
    variables = JuMP.all_variables(gm.model)
    dof = degrees_of_freedom(gm)
    # The "dependent system" contains the dependent variables, y, and the "defining
    # equations", g(x, y) = 0. I want to _exclude_ these equations from this model.
    dependent = get_dependent_system(gm)
    varset = Set(vcat(dof, dependent.variables))
    # We track "other variables" that don't appear in g separately. This is because
    # we don't want to give them a dense derivative. These other variables may
    # not even appear in the "residual constraints", but for now, I don't see any
    # reason to subdivide them further.
    other_variables = filter(∉(varset), variables)
    JuMP.@objective(gm.model, Min, 0)
    # We need to delete these "defining equations" before looking for equations
    # that contain y.
    JuMP.delete.(gm.model, dependent.equations)
    igraph = MPIN.IncidenceGraphInterface(gm.model; include_inequality = true)
    constraints_with_y = Set(JuMP.ConstraintRef[])
    for y in dependent.variables
        for con in MPIN.get_adjacent(igraph, y)
            push!(constraints_with_y, con)
        end
    end
    constraints = JuMP.all_constraints(gm.model; include_variable_in_set_constraints = true)
    constraints_without_y = filter(∉(constraints_with_y), constraints)
    JuMP.delete.(gm.model, constraints_without_y)
    nlp = NLP(gm.model)
    constraints = collect(constraints_with_y)
    return (;
        model = gm.model,
        x = dof,
        y = dependent.variables,
        z = other_variables,
        # NOTE: Order of these constraints is undefined (but run-to-run deterministic)
        # I don't think these need to be consistent with any other vector.
        # I don't think we even need to return these constraints.
        # These are just the constraints in the NLP...
        #constraints,
        nlp,
        x_indices = indices(nlp, dof),
        y_indices = indices(nlp, dependent.variables),
        z_indices = indices(nlp, other_variables),
        # I don't think there is any reason to reorder the NLP's constraints...
        #con_indices = indices(nlp, constraints),
    )
end

function implicit_function_formulation(data::Dict; hessian = true)
    gm = GasModels.instantiate_model(data, GasModels.WPGasModel, GasModels.build_ogf)
    variables = JuMP.all_variables(gm.model)
    dependent = get_dependent_system(gm)
    # TODO: Assert that dependent variables don't appear in the objective
    dof = degrees_of_freedom(gm)
    varset = Set(vcat(dof, dependent.variables))
    other_variables = filter(∉(varset), variables)
    JuMP.delete.(gm.model, dependent.equations)
    igraph = MPIN.IncidenceGraphInterface(gm.model; include_inequality = true)
    constraints_with_y = Set(JuMP.ConstraintRef[])
    for y in dependent.variables
        for con in MPIN.get_adjacent(igraph, y)
            push!(constraints_with_y, con)
        end
    end
    JuMP.delete.(gm.model, constraints_with_y)
    JuMP.delete.(gm.model, dependent.variables)

    # This is the convention we use for the VNO variables
    vno_variables = vcat(dof, other_variables)
    vno_input_dim = length(vno_variables)

    # Now we define intermediate data structures necessary for the VectorNonlinearOracle
    implicit = implicit_model(data)
    residual = residual_model(data)
    ipopt_square = JuMP.optimizer_with_attributes(
        Ipopt.Optimizer,
        "linear_solver" => "ma27",
        #"tol" => 1e-6,
        #"acceptable_tol" => 1e-4,
        "print_level" => 0,
        "print_user_options" => "yes",
        "max_wall_time" => 60.0,
    )
    JuMP.set_optimizer(implicit.model, ipopt_square)
    # Random initialization for good measure
    JuMP.set_start_value.(implicit.y, rand(length(implicit.y)))
    # Define callbacks for implicit functions as closures.
    # TODO: Move these outside this function and parameterize by
    # the `implicit` data structure
    function eval_implicit_function(x)
        JuMP.fix.(implicit.x, x; force = true)
        JuMP.optimize!(implicit.model)
        println(JuMP.termination_status(implicit.model))
        if JuMP.termination_status(implicit.model) != JuMP.LOCALLY_SOLVED
            error(
                """Failed to solve implicit function subproblem.
                Termination status: $(JuMP.termination_status(implicit.model))
                """
            )
        end
        y = JuMP.value.(implicit.y)
        JuMP.set_start_value.(implicit.y, y)
        return y
    end
    function eval_implicit_function_jacobian(x, y)
        all_vars = zeros(length(x) + length(y))
        all_vars[implicit.x_indices] .= x
        all_vars[implicit.y_indices] .= y
        jacobian = eval_constraint_jacobian(implicit.nlp, all_vars)
        ∇yg = jacobian[implicit.eq_indices, implicit.y_indices]
        ∇xg = jacobian[implicit.eq_indices, implicit.x_indices]
        return - ∇yg \ Matrix(∇xg)
    end
    function eval_implicit_constraint(xz)
        nx = length(residual.x)
        nz = length(residual.z)
        # Note that the VectorNonlinearOracle will have to follow this convention.
        x = xz[1:nx]
        z = xz[nx+1:end]
        @assert length(z) == nz
        y = eval_implicit_function(x)
        xyz = zeros(length(x) + length(y) + length(z))
        xyz[residual.x_indices] .= x
        xyz[residual.y_indices] .= y
        xyz[residual.z_indices] .= z
        con_values = eval_constraints(residual.nlp, xyz)
        return con_values
    end
    function eval_implicit_constraint_jacobian(xz)
        nx = length(residual.x)
        nz = length(residual.z)
        x = xz[1:nx]
        z = xz[nx+1:end]
        @assert length(z) == nz
        # TODO: Cache y from eval_implicit_constraint
        y = eval_implicit_function(x)
        xyz = zeros(length(x) + length(y) + length(z))
        xyz[residual.x_indices] .= x
        xyz[residual.y_indices] .= y
        xyz[residual.z_indices] .= z
        jacobian = eval_constraint_jacobian(residual.nlp, xyz)
        ∇xf = jacobian[:, residual.x_indices]
        ∇yf = jacobian[:, residual.y_indices]
        ∇zf = jacobian[:, residual.z_indices]
        ∇xy = eval_implicit_function_jacobian(x, y)
        # We return a dense matrix and a sparse matrix
        return ∇xf + ∇yf * ∇xy, ∇zf
    end
    function eval_implicit_lagrangian_hessian(xz, λ)
        nx = length(residual.x)
        nz = length(residual.z)
        x = xz[1:nx]
        z = xz[nx+1:end]
        @assert length(z) == nz
        y = eval_implicit_function(x)
        xyz = zeros(length(x) + length(y) + length(z))
        xyz[residual.x_indices] .= x
        xyz[residual.y_indices] .= y
        xyz[residual.z_indices] .= z
        xy = zeros(length(x) + length(y))
        xy[implicit.x_indices] .= x
        xy[implicit.y_indices] .= y

        # TODO: These can be retrieved from a cache
        ∇g = eval_constraint_jacobian(implicit.nlp, xy)
        ∇yg = ∇g[implicit.eq_indices, implicit.y_indices]
        ∇f = eval_constraint_jacobian(residual.nlp, xyz)
        ∇yf = ∇f[:, residual.y_indices]

        λg = zeros(length(implicit.nlp.constraints))
        λg[implicit.eq_indices] .= - ∇yg' \ (∇yf' * λ)
        ∇2g_implicit = eval_lagrangian_hessian(implicit.nlp, xy, λg)
        ∇2f = eval_lagrangian_hessian(residual.nlp, xyz, λ)

        # We "expand" ∇2g into the x,y,z space to match the residual function
        ∇2g = SparseArrays.spzeros(length(xyz), length(xyz))
        xy_ind_implicit = vcat(implicit.x_indices, implicit.y_indices)
        xy_ind_residual = vcat(residual.x_indices, residual.y_indices)
        ∇2g[xy_ind_residual, xy_ind_residual] = ∇2g_implicit[xy_ind_implicit, xy_ind_implicit]

        # Similarly, we need to expand ∇xy
        ∇xy_implicit = eval_implicit_function_jacobian(x, y)
        ∇xy = zeros(length(y), length(xz))
        # The convention is that x uses indices 1:nx
        ∇xy[:, 1:nx] .= ∇xy_implicit

        # Here, we compress x and z into "x" to limit the number of matrices
        # we need to track.
        # This order (x,z) is the convention we use to construct the VNO.
        xz_indices = vcat(residual.x_indices, residual.z_indices)
        ∇xxf = ∇2f[xz_indices, xz_indices]
        ∇xyf = ∇2f[xz_indices, residual.y_indices]
        ∇yyf = ∇2f[residual.y_indices, residual.y_indices]

        # ∇2g has been projected up into the space of the residual function
        ∇xxg = ∇2g[xz_indices, xz_indices]
        ∇xyg = ∇2g[xz_indices, residual.y_indices]
        ∇yyg = ∇2g[residual.y_indices, residual.y_indices]

        # ∇xy is now ordered in the space of x,z in the residual model. We need
        # to make sure that the order matches the (x,z) convention used by the
        # VNO (and therefore the Hessian matrices we compute)
        #∇xy = ∇xy[:, xz_indices]

        Hxy = (∇xyf + ∇xyg) * ∇xy
        hessian = (
            ∇xxf + ∇xxg
            + Hxy + Hxy'
            + ∇xy' * (∇yyf + ∇yyg) * ∇xy
        )
        return hessian
    end

    # Define the Jacobian structure
    nx = length(residual.x)
    nrow = length(residual.nlp.constraints)
    # Note that we arrange this dense matrix's entries in column-major order
    IJx = [(i, j) for j in 1:nx for i in 1:nrow]
    # Evaluate ∇zf with a dummy input to get its sparsity structure, which we assume is constant
    ∇zf = eval_constraint_jacobian(residual.nlp, ones(length(residual.nlp.variables)))[:, residual.z_indices]
    Iz, Jz, _ = SparseArrays.findnz(∇zf)
    IJz = collect(zip(Iz, Jz .+ nx))
    jacobian_structure = vcat(IJx, IJz)

    # Column-major Hessian structure assuming a dense Hessian. The Hessian
    # is not necessarily entirely dense here but exploiting sparsity
    # gets complicated quickly.
    hessian_lagrangian_structure = [(i, j) for j in 1:vno_input_dim for i in 1:vno_input_dim]
    hessian_triu_mask = map(ij -> (ij[1] <= ij[2]), hessian_lagrangian_structure)
    # We will apply the above mask to the flattened dense Hessian matrix
    hessian_lagrangian_structure = hessian_lagrangian_structure[hessian_triu_mask]
    if hessian
        eval_hessian_lagrangian = (ret, xz, λ) -> begin
            ∇2f = eval_implicit_lagrangian_hessian(xz, λ)
            hnnz = length(hessian_lagrangian_structure)
            ret[1:hnnz] .= reshape(∇2f, :)[hessian_triu_mask]
            return
        end
    else
        # Recall that we disable Hessian evaluation (and thus use LBFGS) by
        # omitting this callback (or setting it to nothing).
        eval_hessian_lagrangian = nothing
    end

    VNO = MOI.VectorNonlinearOracle(;
        dimension = vno_input_dim,
        l = residual.nlp.constraint_lbs,
        u = residual.nlp.constraint_ubs,
        eval_f = (ret, x) -> begin
            ret .= eval_implicit_constraint(x)
            return
        end,
        jacobian_structure,
        eval_jacobian = (ret, xz) -> begin
            ∇xf, ∇zf = eval_implicit_constraint_jacobian(xz)
            # Note that ∇xf is a dense matrix
            ret[1:(nx*nrow)] .= reshape(∇xf, :)
            z_nnz = SparseArrays.nnz(∇zf)
            ret[(nx*nrow+1):(nx*nrow+z_nnz)] .= ∇zf.nzval
            return
        end,
        hessian_lagrangian_structure,
        eval_hessian_lagrangian,
    )
    JuMP.@constraint(gm.model, vno_variables in VNO)
    return gm, (;
        implicit,
        residual,
        VNO,
        x = vno_variables,
    )
end

if false
    # Here's a case that works
    #data = NaermLib.pipeline_model("TallgrassIGT")

    # Here's a case that kind of converges, but IPOPT reports infeasible???
    #data = NaermLib.pipeline_model("KernRiverGT-CUI")

    # Here's a case that doesn't work (yet!)
    #data = NaermLib.pipeline_model("NGPCA")

    #data = NaermLib.pipeline_model("Portland")
    #data = NaermLib.pipeline_model("Viking", "v1")
    #data = NaermLib.pipeline_model("MississippiRiver")

    file = joinpath(dirname(dirname(pathof(GasModels))), "test", "data", "matgas", "case-6.m")
    data = GasModels.parse_file(file)
    _correct_data!(data)
    ipopt = JuMP.optimizer_with_attributes(
        Ipopt.Optimizer,
        "linear_solver" => "ma57",
        #"ma57_pivot_order" => 4,
        "tol" => 1e-6,
        "acceptable_tol" => 1e-4,
        "print_level" => 5,
        "print_user_options" => "yes",
        #"max_wall_time" => 60.0,
    )
    gm, _ = implicit_function_formulation(data; hessian = true)
    JuMP.set_optimizer(gm.model, ipopt)
    JuMP.optimize!(gm.model)
end
