import dolfin as df

#######################################################

def rectangle_2d(lx, ly, source_rad, elements_x, elements_y, type=None):
    """Prepares 2D geometry. Returns facet function with 1, 2 on parts of
    the boundary."""
    # Define 2d rectangle based on width and high
    x0 = 0.0
    x1 = x0 + lx
    y0 = 0.0
    y1 = y0 + ly
    if type == None:
        mesh = df.RectangleMesh(df.Point(x0, y0), df.Point(x1, y1), elements_x, elements_y)
    else:
        mesh = df.RectangleMesh(df.Point(x0, y0), df.Point(x1, y1), elements_x, elements_y, type)
    # Extract edges of boundary
    boundary_parts = df.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    left = df.AutoSubDomain(lambda x: df.near(x[0], x0))
    right = df.AutoSubDomain(lambda x: df.near(x[0], x1))
    down = df.AutoSubDomain(lambda x: df.near(x[1], y0))
    up = df.AutoSubDomain(lambda x: df.near(x[1], y1))

    left.mark(boundary_parts, 1)
    right.mark(boundary_parts, 2)
    up.mark(boundary_parts, 3)
    down.mark(boundary_parts, 4)

    #    class Source(df.SubDomain):
    #        def inside(self, x, on_boundary):
    #            radius = source_rad
    #            return True if (x[0] * x[0] + x[1] * x[1]) < radius*radius else False
    #
    #    mf = df.CellFunctionSizet(mesh, 0)
    #    Source().mark(mf, 1)

    return boundary_parts

def nonlinvarsolver(F, w, bcs, newton_rel, newton_abs, solver_type):
    J = df.derivative(F, w)
    # Initialize solver
    problem = df.NonlinearVariationalProblem(F, w, bcs=bcs, J=J)
    solver = df.NonlinearVariationalSolver(problem)
    solver.parameters['newton_solver']['relative_tolerance'] = newton_rel
    solver.parameters['newton_solver']['absolute_tolerance'] = newton_abs
    solver.parameters['newton_solver']['linear_solver'] = solver_type
    return solver

def write_field2output(outputfile, field, fieldname, timestep):
    field.rename(fieldname, fieldname)
    outputfile.write(field, timestep)

def F(u):
    return df.Identity(len(u)) + df.grad(u)

def B(u):
    return F(u) * F(u).T

def C(u):
    return F(u).T * F(u)

def E(u):
    return 0.5 * (C(u) - df.Identity(len(u)))

def E_lin(u):
    return df.sym(df.grad(u))

def J(u):
    return df.det(F(u))

def J_lin(u):
    return 1.0 + df.tr(E_lin(u))

def KS(u):
    return 0.5 * (B(u) - df.Identity(len(u)))

def T_E(u, nS, mu, lambdaS):
     return 2 * mu * KS(u) / J(u) + lambdaS * (1 - nS) * (1 - nS) * ((1 / (1 - nS)) - (1 / (J(u) - nS))) * df.Identity(len(u))
    #return 2.0 * mu * E_lin(u) + lambdaS * df.tr(E_lin(u)) * df.Identity(len(u))

def T(u, p, nS, muS, lambdaS):
    return T_E(u, nS, muS, lambdaS) - p * df.Identity(len(u))
    #return T_E(u, n, mu, lmbda) - p * (J_lin(u) * df.Identity(len(u)) - 2.0 * E_lin(u))

#################################################

def run_luis():

    # Geometry
    lx = 2
    ly = 2
    elements = 25
    
    # Time discretisation
    T_end = 5
    dt = 0.1
    
    # Material (healthy Solid)
    lambdaSh = 3312
    muSh = 662
    rhoShR = 1190.0
    nSh0 = 0.6
    
    # Material (tumorous Solid)
    lambdaSt = 3312
    muSt = 662
    rhoStR = 1190.0
    nSt0 = 0.15
    
    # Material (interstitial Fluid)
    rhoFR = 993.3
    muF = 0.0007
    KF = 1.0E-6 #5E-13 oder 5E-8
    
    MFn = 80000     #0.18
    DFn = 1.0E-3    #6.6E-10
    alpha_Fn1 = -5.33E-10
    alpha_Fn2 = -5.33E-11
    cn_tres = 1.6
    
    MFt = 1.3E13
    DFt = 1.0E-3
    kFt = 0.3
    NFt = 10E11
    ct_tres = df.Constant(1.5)
    
    MFv = 50
    DFv = 1.0E-2    #4E-11
    alpha_Fv = 15
    
    MFd = 93
    DFd = 5E-9
    
    # Source geometry
    x_source = lx/2
    y_source = ly/2
    radius_source = 0.2
    
    ct0 = 1
    cn0 = 2
    
    b = df.Constant([0.0, 0.0])
    
    # Solution algorithm
    solv_sType = "mumps"
    solv_nTolRel = 1E-7
    solv_nTolAbs = 1E-8
    
    ######################################################
    
    # Solution output
    output_file = df.XDMFFile("solution/10_Production_Solid.xdmf")
    output_file.parameters["flush_output"] = True
    output_file.parameters["functions_share_mesh"] = True
        
    facet_function = rectangle_2d(lx, ly, radius_source, lx * elements, ly * elements)
    mesh = facet_function.mesh()
    n = df.FacetNormal(mesh)
    
    dx = df.Measure("dx")
    ds1 = df.Measure("ds", subdomain_data=facet_function, subdomain_id=1)  # Declares and defines left side
    ds2 = df.Measure("ds", subdomain_data=facet_function, subdomain_id=2)  # Declares and defines right side
    ds3 = df.Measure("ds", subdomain_data=facet_function, subdomain_id=3)  # Declares and defines top side
    ds4 = df.Measure("ds", subdomain_data=facet_function, subdomain_id=4)  # Declares and defines bottom side
    
    V1 = df.FunctionSpace(mesh, "P", 1)
    V2 = df.VectorFunctionSpace(mesh, "P", 1)
    V3 = df.TensorFunctionSpace(mesh, "P", 1)
    
    # Build function space
    element_u = df.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_p = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_cn = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_ct = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_cv = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_cd = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = df.MixedElement([element_u, element_p, element_cn, element_ct, element_cv, element_cd])
    W = df.FunctionSpace(mesh, TH)
    
    # Get Ansatz and test functions
    w = df.Function(W)
    _w = df.TestFunction(W)
    w_n = df.Function(W)
    u, p, cn, ct, cv, cd = df.split(w)
    _u, _p, _cn, _ct, _cv, _cd = df.split(_w)
    u_n, p_n, cn_n, ct_n, cv_n, cd_n = df.split(w_n)
    
    #######################################################
    
    bc_u1 = df.DirichletBC(W.sub(0).sub(0), df.Constant(0), facet_function, 1)  # ux=0 auf linkem Rand
    bc_u2 = df.DirichletBC(W.sub(0).sub(0), df.Constant(0), facet_function, 2)  # ux=0 auf rechtem Rand
    bc_u3 = df.DirichletBC(W.sub(0).sub(1), df.Constant(0), facet_function, 3)  # uy=0 auf dem unteren Rand
    bc_u4 = df.DirichletBC(W.sub(0).sub(1), df.Constant(0), facet_function, 4)  # uy=0 auf dem oberen Rand
    
    bc_cn1 = df.DirichletBC(W.sub(2), df.Constant(cn0), facet_function, 1)  # cn0  auf dem linken Rand
    bc_cn2 = df.DirichletBC(W.sub(2), df.Constant(cn0), facet_function, 2)  # cn0  auf dem rechten Rand
    bc_cn3 = df.DirichletBC(W.sub(2), df.Constant(cn0), facet_function, 3)  # cn0  auf dem unteren Rand
    bc_cn4 = df.DirichletBC(W.sub(2), df.Constant(cn0), facet_function, 4)  # cn0  auf dem oberen Rand
    
    d_bound = [bc_u1, bc_u2, bc_u3, bc_u4, bc_cn1, bc_cn2, bc_cn3, bc_cn4]
    
    #######################################################
    
    tt = df.Function(V1)
    Fn_prod = df.Function(V1)
    Ft_prod = df.Function(V1)
    Fv_prod = df.Function(V1)
    Fd_prod = df.Function(V1)
    
    F_prod = df.Function(V1)
    S_prod = df.Function(V1)
    
    tt.assign(df.project(df.Constant(0.0), V1))
    Ft_prod.assign(df.project(df.conditional(df.le(ct, ct_tres), df.project(ct, V1) * MFt * kFt, 0.0),V1))
    Fn_prod.assign(df.project(df.conditional(df.lt(ct, ct_tres), alpha_Fn1 * (MFt * df.project(ct, V1) + Ft_prod), alpha_Fn2 * (MFt * df.project(ct, V1) + Ft_prod)), V1))
    Fv_prod.assign(df.project(df.conditional(df.le(cn, cn_tres), df.conditional(df.le(ct, ct_tres), df.project(ct, V1) * alpha_Fv, 0.0), 0.0), V1))
    Fd_prod.assign(df.project(df.Constant(0.0), V1))
    
    F_prod.assign(df.project(df.Constant(-1e-8), V1))
    S_prod.assign(df.project(-F_prod, V1))   #?
    
    nSt_n = df.Function(V1)
    nSt_n.assign(df.project(nSt0, V1))
    nSt = nSt0 * df.exp((100 /(rhoStR * nSt_n)) * tt) / J(u)
    nSh = nSh0 / J(u)
    nS0 = nSh0 + nSt0
    nS = nSh + nSt
    nF = 1 - nSh - nSt
    
    rho = rhoShR * nSh + rhoStR * nSt + rhoFR * nF
    muS = muSh + muSt
    lambdaS = lambdaSt+lambdaSh
    
    
    dt_trE = (1 / dt) * df.tr(E(u) - E(u_n))
    dt_nS = (S_prod / rho) - nF * dt_trE
    dt_nF = - dt_nS
    dt_cn = (1 / dt) * (cn - cn_n)
    dt_ct = (1 / dt) * (ct - ct_n)
    dt_cv = (1 / dt) * (cv - cv_n)
    dt_cd = (1 / dt) * (cd - cd_n)
    #wF = -KF * df.dot(df.grad(p), df.Identity(len(u)) * J(u) - 2.0 * E(u))
    wF = -KF * df.grad(p)
    nFwF = - (KF / muF) * (df.grad(p) - rhoFR * b)
    nFcFnwFn = -DFn * df.grad(cn) + nF * cn * wF
    nFcFtwFt = -DFt * df.grad(ct) + nF * ct * wF
    nFcFvwFv = -DFv * df.grad(cv) + nF * cv * wF
    nFcFdwFd = -DFd * df.grad(cd) + nF * cd * wF
    
    #######################################################
    
    balance_mass = (dt_nF + nF * dt_trE) * _p * dx - df.inner(nFwF, df.grad(_p)) * dx - (F_prod / rhoFR) * _p * dx   # Massenbilanz mit Produktionsterm rechnet, aber Ergebnis fragwürdig
    balance_momentum = df.inner(T(u, p, nS, muS, lambdaS), df.grad(_u)) * dx - rho * df.inner(b, _u) * dx
    cmFn_concentration = (nF * dt_cn + cn * dt_trE) * _cn * dx - df.inner(nFcFnwFn, df.grad(_cn)) * dx - (Fn_prod/MFn + cn * (S_prod/rhoStR)) * _cn * dx
    cmFt_concentration = (nF * dt_ct + ct * dt_trE) * _ct * dx - df.inner(nFcFtwFt, df.grad(_ct)) * dx - (Ft_prod/MFt + ct * (S_prod/rhoStR)) * _ct * dx
    cmFv_concentration = (nF * dt_cv + cv * dt_trE) * _cv * dx - df.inner(nFcFvwFv, df.grad(_cv)) * dx - (Fv_prod/MFv + cv * (S_prod/rhoStR)) * _cv * dx
    cmFd_concentration = (nF * dt_cd + cd * dt_trE) * _cd * dx - df.inner(nFcFdwFd, df.grad(_cd)) * dx - (Fd_prod/MFd + cd * (S_prod/rhoStR)) * _cd * dx
    
    res_tot = balance_momentum + balance_mass + cmFn_concentration + cmFt_concentration + cmFv_concentration + cmFd_concentration
    
    # Define problem solution
    solver = nonlinvarsolver(res_tot, w, d_bound, solv_nTolRel, solv_nTolAbs, solv_sType)
    
    #######################################################
    
    # Initialize solution time
    t = 0.0
    
    u_init = df.Constant([0.0, 0.0])
    p_init = df.Constant(0.0)
    cn_init = df.Constant(cn0)
    ct_init = df.Expression(("ct0*exp(-a*(pow((x[0]-x_source),2)+pow((x[1]-y_source),2)))"), degree=2, ct0=ct0, a=100, x_source=x_source, y_source=y_source)
    cv_init = df.Constant(0.0)
    cd_init = df.Constant(0.0)
    
    u_n = df.interpolate(u_init, V2)
    p_n = df.interpolate(p_init, V1)
    cn_n = df.interpolate(cn_init, V1)
    ct_n = df.interpolate(ct_init, V1)
    cv_n = df.interpolate(cv_init, V1)
    cd_n = df.interpolate(cd_init, V1)
    
    write_field2output(output_file, u_n, "u", t)
    write_field2output(output_file, p_n, "p", t)
    write_field2output(output_file, cn_n, "c_m^Fn", t)
    write_field2output(output_file, ct_n, "c_m^Ft", t)
    write_field2output(output_file, cv_n, "c_m^Fv", t)
    write_field2output(output_file, cd_n, "c_m^Fd", t)
    write_field2output(output_file, df.project(T(u_n, p_n, nS, lambdaS, muS), V3), "Stress", t)
    write_field2output(output_file, df.project(E_lin(u_n), V3), "Strain", t)
    write_field2output(output_file, df.project(wF, V2), "w_F", t)
    write_field2output(output_file, df.project(nSh0, V1), "nSh", t)
    write_field2output(output_file, df.project(nSt0, V1), "nSt", t)
    write_field2output(output_file, df.project(nF, V1), "nF", t)
    
    #df.assign(w_n.sub(0), u_n)
    df.assign(w_n.sub(1), p_n)
    df.assign(w_n.sub(2), cn_n)
    df.assign(w_n.sub(3), ct_n)
    df.assign(w_n.sub(4), cv_n)
    df.assign(w_n.sub(5), cd_n)
    
    t = t + dt
    
    #######################################################
    
    # Time loop
    while t <= T_end:
        # Print current time
        df.info("Time: {}".format(t))
    
        # Calculate current solution
        solver.solve()
    
        # Output solution
        u, p, cn, ct, cv, cd = w.split()
    
        tt.assign(df.project(df.conditional(df.ge(ct, ct_tres), tt + dt, 0.0), V1))
        Ft_prod.assign(df.project(df.conditional(df.le(ct, ct_tres), df.project(ct, V1) * MFt * kFt, 0.0), V1))
        Fn_prod.assign(df.project(df.conditional(df.lt(ct, ct_tres), alpha_Fn1 * (MFt * df.project(ct, V1) + Ft_prod), alpha_Fn2 * (MFt * df.project(ct, V1) + Ft_prod)), V1))
        Fv_prod.assign(df.project(df.conditional(df.le(cn, cn_tres), df.conditional(df.lt(ct, ct_tres), df.project(ct, V1) * alpha_Fv, 0.0), 0.0), V1))
        Fd_prod.assign(df.project(df.Constant(0.0), V1))
    
        F_prod.assign(df.project(df.project(df.Constant(1e-8), V1)))
        S_prod.assign(df.project(-F_prod, V1))  # ?
    
        nSt = nSt0 * df.exp((100 / (rhoStR * nSt_n)) * tt) / J(u)
        nF = 1 - nSh - nSt
    
        write_field2output(output_file, u, "u", t)
        write_field2output(output_file, p, "p", t)
        write_field2output(output_file, cn, "c_m^Fn", t)
        write_field2output(output_file, ct, "c_m^Ft", t)
        write_field2output(output_file, cv, "c_m^Fv", t)
        write_field2output(output_file, cd, "c_m^Fd", t)
        write_field2output(output_file, df.project(T(u, p, nS, lambdaS, muS), V3), "Stress", t)
        write_field2output(output_file, df.project(E_lin(u), V3), "Strain", t)
        write_field2output(output_file, df.project(wF, V2), "w_F", t)
        write_field2output(output_file, df.project(nSh, V1), "nSh", t)
        write_field2output(output_file, df.project(nSt, V1), "nSt", t)
        write_field2output(output_file, df.project(nF, V1), "nF", t)
    
        # Update history fields
        w_n.assign(w)
        nSt_n.assign(df.project(nSt, V1))
        # Increment solution time
        t = t + dt