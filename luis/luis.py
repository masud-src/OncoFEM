import dolfin as df
from pathlib import Path
import numpy as np
import meshio, os
from ufl import *
#######################################################

class InitialConditions(df.UserExpression):
    def __init__(self, params,  **kwargs):
        super().__init__(**kwargs)
        self.params = params
    def eval_cell(self, values, x, ufc_cell):
        values[0] = 0.0         # ux = 0
        values[1] = 0.0         # uy = 0
        values[2] = 0.0         # uz = 0
        values[3] = 0.0         # p = 0
        values[4] = 0.0         # nSt = 0
        values[5] = 0.0         # cFn = 0
        values[6] = 0.0         # cFt = 0
        values[7] = 0.0         # cFv = 0
        if (x[0] - self.params.x_infusion) * (x[0] - self.params.x_infusion) + (x[1] - self.params.y_infusion) * (x[1] - self.params.y_infusion) <= pow(self.params.radius_infusion, 2):
            values[8] = self.params.cFd_infusion_concentration
        else:
            values[8] = 0.0

    def value_shape(self):
        return (9,)

class Parameter():
    def __init__(self):
        pass

def generatecirclemesh(name, cp, ms):
    f = open(name, "w+")
    f.write("// Gmsh project created on Fri Sep 16 12:56:59 2022""\n""SetFactory(\"OpenCASCADE\");""\n""//+""\n")
    f.write("Point(1) = {0, 1, 0, " + repr(ms) + "};\n")
    f.write("Point(2) = {0, 0, 0, " + repr(ms) + "};\n")
    f.write("Point(3) = {1, 0, 0, " + repr(ms) + "};\n")
    f.write("Circle(1) = {1, 2, 3};\n")
    f.write("Line(2) = {3, 2};\n")
    f.write("Line(3) = {2, 1};\n")
    f.write("Curve Loop(1) = {1, 2, 3};\n")
    f.write("Plane Surface(1) = {1};\n")
    f.write("Physical Surface(3) = {1};\n")
    f.write("Physical Curve(1) = {2};\n")   # bottom
    f.write("Physical Curve(2) = {3};\n")   # left
    #f.write("Point {2} In Surface {1};\n")
    #f.write("Recombine Surface {4};\n")
    f.write("MeshSize {2} = " + repr(cp) + ";")

def getXDMF(inputdirectory):
    """
    Gathers all needed input files from a respective folder in workingdata environment and returns
    the files in the following order: (tetra.xdmf), triangle.xdmf, lines.xmdf

    *Arguments:*
        inputdirectory: input_folder

    *Example:*
        getXDMF("Terzaghi_2d")
    """
    try:
        xdmf_files = []
        if os.path.isfile("workingdata/" + inputdirectory+"/tetra.xdmf"):
            mesh = df.Mesh()
            with df.XDMFFile("workingdata/" + inputdirectory + "/tetra.xdmf") as infile:
                infile.read(mesh)

            tetra_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with df.XDMFFile("workingdata/" + inputdirectory + "/tetra.xdmf") as infile:
                infile.read(tetra_mvc, "name_to_read")
            tetra = df.MeshFunction("size_t", mesh, tetra_mvc)
            xdmf_files.append(tetra)

            triangle_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with df.XDMFFile("workingdata/" + inputdirectory + "/triangle.xdmf") as infile:
                infile.read(triangle_mvc, "name_to_read")
            triangle = df.MeshFunction("size_t", mesh, triangle_mvc)
            xdmf_files.append(triangle)

            if os.path.isfile("workingdata/" + inputdirectory+"/line.xdmf"):
                line_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
                with df.XDMFFile("workingdata/" + inputdirectory + "/line.xdmf") as infile:
                    infile.read(line_mvc, "name_to_read")
                line = df.MeshFunction("size_t", mesh, line_mvc)
                xdmf_files.append(line)

        else:
            mesh = df.Mesh()
            with df.XDMFFile("workingdata/" + inputdirectory + "/triangle.xdmf") as infile:
                infile.read(mesh)

            triangle_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with df.XDMFFile("workingdata/" + inputdirectory + "/triangle.xdmf") as infile:
                infile.read(triangle_mvc, "name_to_read")
            triangle = df.MeshFunction("size_t", mesh, triangle_mvc)
            xdmf_files.append(triangle)

            line_mvc = df.MeshValueCollection("size_t", mesh, mesh.topology().dim())
            with df.XDMFFile("workingdata/" + inputdirectory + "/line.xdmf") as infile:
                infile.read(line_mvc, "name_to_read")
            line = df.MeshFunction("size_t", mesh, line_mvc)
            xdmf_files.append(line)

        return xdmf_files
    except:
        print("input not working")
    pass

def msh2xdmf(inputfile, outputfolder):
    """
    Generates from input msh file two or three output files in xdmf format for input into FEniCS. Output files are:
    (tetra.xdmf), triangle.xdmf, lines.xmdf

    *Arguments:*
        inputfile: input_msh_file
        outputfolder: output_folder

    *Example:*
        msh2xdmf("inputdata/Terzaghi.msh", "Terzaghi_2d")
    """
    Path("workingdata/" + outputfolder).mkdir(parents=True, exist_ok=True)
    msh = meshio.read(inputfile)
    line_cells = []
    triangle_cells = []
    tetra_cells = []
    for cell in msh.cells:
        if cell.type == "tetra":
            if len(tetra_cells) == 0:
                tetra_cells = cell.data
            else:
                tetra_cells = np.vstack([tetra_cells, cell.data])
        elif cell.type == "triangle":
            if len(triangle_cells) == 0:
                triangle_cells = cell.data
            else:
                triangle_cells = np.vstack([triangle_cells, cell.data])
        elif cell.type == "line":
            if len(line_cells) == 0:
                line_cells = cell.data
            else:
                line_cells = np.vstack([line_cells, cell.data])

    line_data = []
    triangle_data = []
    tetra_data = []
    for key in msh.cell_data_dict["gmsh:physical"].keys():
        if key == "line":
            if len(line_data) == 0:
                line_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                line_data = np.vstack([line_data, msh.cell_data_dict["gmsh:physical"][key]])
        elif key == "triangle":
            if len(triangle_data) == 0:
                triangle_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                triangle_data = np.vstack([triangle_data, msh.cell_data_dict["gmsh:physical"][key]])
        elif key == "tetra":
            if len(tetra_data) == 0:
                tetra_data = msh.cell_data_dict["gmsh:physical"][key]
            else:
                tetra_data = np.vstack([tetra_data, msh.cell_data_dict["gmsh:physical"][key]])
    if len(tetra_cells) != 0:
        print("write tetra_mesh")
        tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells},
                                 cell_data={"name_to_read": [tetra_data]})
        meshio.write("workingdata/" + outputfolder + "/tetra.xdmf", tetra_mesh)
    if len(triangle_cells) != 0:
        print("write triangle_mesh")
        triangle_mesh = meshio.Mesh(points=msh.points, cells={"triangle": triangle_cells},
                                    cell_data={"name_to_read": [triangle_data]})
        meshio.write("workingdata/" + outputfolder + "/triangle.xdmf", triangle_mesh)
    if len(line_cells) != 0:
        print("write line_mesh")
        line_mesh = meshio.Mesh(points=msh.points, cells=[("line", line_cells)],
                                cell_data={"name_to_read": [line_data]})
        meshio.xdmf.write("workingdata/" + outputfolder + "/line.xdmf", line_mesh)
    pass

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

def calcStress_vonMises(T):
    """
    calculates scalar von Mises stress

    *Arguments:*
    T: Stress tensor (2D/3D)

    *Example:*
    calcStress_vonMises(T)
    """
    sig_x = T[0, 0]
    sig2_x = sig_x * sig_x
    sig_y = T[1, 1]
    sig2_y = sig_y * sig_y
    tau_xy = T[1, 0] * T[1, 0]
    if shape(T)[0] == 2:
        return sqrt(sig2_x + sig2_y - sig_x * sig_y + 3.0 * tau_xy)
    elif shape(T)[0] == 3:
        sig_z = T[2, 2]
        sig2_z = sig_z * sig_z
        tau_xz = T[0, 2] * T[0, 2]
        tau_yz = T[1, 2] * T[1, 2]
        return sqrt(sig2_x + sig2_y + sig2_z - sig_x * sig_y - sig_x * sig_z - sig_y * sig_z + 3.0 * (
                        tau_xy + tau_xz + tau_yz))

def gen_mesh(file, centerparameter, meshsize, outputfile):
    # Generate mesh
    generatecirclemesh(file, centerparameter, meshsize)
    os.system("gmsh -2 workingdata/Mesh.geo -o workingdata/Mesh.msh")
    # Solution output
    outputfile.parameters["flush_output"] = True                   # file can be filled with multiple fields
    outputfile.parameters["functions_share_mesh"] = True           # fields in file use the same mesh
    # Load mesh
    inputfile = "workingdata/Mesh.msh"
    outputfolder = "Mesh"
    msh2xdmf(inputfile, outputfolder)
    domain, facet_function = getXDMF(outputfolder)
    mesh = facet_function.mesh()
    return mesh, facet_function

def function_space(mesh):
    # Build function space
    element_u = df.VectorElement("CG", mesh.ufl_cell(), 2)
    element_p = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    element_nSt = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    element_cn = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    element_ct = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    element_cv = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element_cd = df.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    TH = df.MixedElement([element_u, element_p, element_nSt, element_cn, element_ct, element_cv, element_cd])
    return df.FunctionSpace(mesh, TH), df.FunctionSpace(mesh, "P", 1), df.VectorFunctionSpace(mesh, "P", 1)

def solve(W, V1, V2, bcs, params):
    prm = df.parameters["form_compiler"]
    prm["quadrature_degree"] = 2
    phase = "fluid tumor growth"
    status = "undetected"
    dx = df.Measure("dx")
    # Get Ansatz and test functions
    w = df.Function(W)
    _w = df.TestFunction(W)
    w_n = df.Function(W)
    u, p, nSt, cFn, cFt, cFv, cFd = df.split(w)
    _u, _p, _nSt, _cFn, _cFt, _cFv, _cFd = df.split(_w)
    u_n, p_n, nSt_n, cFn_n, cFt_n, cFv_n, cFd_n = df.split(w_n)
    #######################################################
    # Kinematik
    #######################################################
    I = df.Identity(len(u))
    F_S = I + df.grad(u)
    F_Sn = I + df.grad(u_n)
    B_S = F_S * F_S.T
    C_S = F_S.T * F_S
    J_S = df.det(F_S)
    dFdt = (F_S - F_Sn) / params.dt
    L_S = dFdt * df.inv(F_S)
    D_S = (L_S + L_S.T) / 2.0
    K_S = (B_S - I) / 2.0

    nSh = params.nSh0 / J_S
    nS = nSh + nSt
    nF = 1.0 - nS

    muS = (params.muSh * nSh + params.muSt * nSt) / nS
    lambdaS = (params.lambdaSt * nSt + params.lambdaSh * nSh) / nS
    rhoS = (params.rhoShR * nSh + params.rhoStR * nSt) / nS

    T_E = (2 * K_S * muS / J_S) + lambdaS * (1 - params.nS0) * (1 - params.nS0) * ((1/(1-params.nS0)) - (1/(J_S-params.nS0))) * I
    #T_E = (muS * (B_S - I) + lambdaS * ln(J_S) * I) / J_S
    #T_E = 1/J_SE * (muS * (B_SE - I) + lambdaS * ln(J_SE) * I)
    T = T_E - p * I
    P = J_S * T * df.inv(F_S).T

    ########################################
    # Production terms
    ########################################
    breakout = df.Function(V1)
    breakout.assign(df.project(df.Constant(0.0), V1))
    cFn_VEGF_touch = df.Function(V1)
    cFn_VEGF_touch.assign(df.project(df.Constant(0.0), V1))
    nSt_detection = df.Function(V1)
    nSt_detection.assign(df.project(df.Constant(0.0), V1))
    nSt_begin = df.Function(V1)
    nSt_begin.assign(df.project(df.Constant(0.0), V1))
    angio_begin = df.Function(V1)
    angio_begin.assign(df.project(df.Constant(0.0), V1))
    angio_activate = df.Function(V1)
    angio_activate.assign(df.project(df.Constant(0.0), V1))

    nSt_Prod = df.Function(V1)

    rhoFn_Prod = df.Function(V1)
    rhoFt_Prod = df.Function(V1)
    rhoFv_Prod = df.Function(V1)
    rhoFd_Prod = df.Function(V1)

    ###################
    H1 = df.conditional(df.gt(cFn, params.cFn_min_growth), 1.0, 0.0)
    H2 = df.conditional(df.le(cFn, params.cFN_min_necrosis), 1.0, 0.0)
    H3 = df.conditional(df.gt(cFt, params.cFt_Ft2nSt), 1.0, 0.0)
    H4 = df.conditional(df.ge(cFd, params.cFd_min_impact), 1.0, 0.0)
    H5 = df.conditional(df.ge(cFv, 0.9875 * params.cFv_max), 0.0, 1.0)
    H6 = df.conditional(df.gt(cFn, params.cFn_min_VEGF_prod), 0.0, 1.0)
    H7 = df.conditional(df.gt(cFv, params.cFv_angio), 1.0, 0.0)
    H8 = df.conditional(df.ge(cFv, params.cFv_init), 1.0, 0.0)
    H9 = df.conditional(df.gt(cFd, 0.0), 1.0, 0.0)

    hat_St_Fn_gain = H1 * abs(nSt) * params.rhoStR * params.v_St_growth * (1.0 - H4) * (H7 + (1 - H7) * ((cFn - params.cFn_min_growth)/(params.Kgr + cFn - params.cFn_min_growth)))
    hat_St_Fn_loss = H2 * abs(nSt) * params.rhoStR * params.v_St_necrosis
    hat_St_Fd_loss = H4 * abs(nSt) * params.rhoStR * (params.v_Fd_max + cFd * params.reaction_rate_cFd)

    hat_Ft_Fn_gain = nF * cFt * params.MFt * H1 * params.kappa_Ft_proliferation * (1.0 - (cFt/params.cFt_threshold)) * (1.0 - H4)
    hat_Ft_Fn_loss = nF * cFt * params.MFt * H2 * params.v_St_necrosis
    hat_Ft_Fd_loss = nF * cFt * params.MFt * H4 * (params.v_Fd_max + cFd * params.reaction_rate_cFd)
    hat_Fn_basal_loss = params.v_In_basal * params.MFn * params.NFt * (nF * cFt * params.MFt + nSt * params.rhoStR)
    hat_Fn_proli_loss = params.f_proli * (hat_Ft_Fn_gain * 1E-4 + hat_St_Fn_gain)
    hat_Fn_angio_gain = params.f_proli * hat_St_Fn_gain * angio_activate
    hat_Fn_regain = nF * params.MFn * cFn * params.v_Fn_regrowth * (1 - (cFn/params.cFn0)) * (H4 + 0.1 * H7)
    hat_Fv_St_gain = nF * abs(nSt) * params.MFv * params.v_Fv_plus * H8 * ((params.cFn0-cFn)/(params.cFn0-cFn+params.cFn_min_growth)) * (1 - (cFv/params.cFv_max)) * H5
    hat_Fd_St_loss = params.v_Fd * (hat_St_Fd_loss + hat_Ft_Fd_loss) * H4 + params.v_Fd_halflife * cFd * params.MFd * H9

    hat_St = (abs(hat_St_Fn_gain) - hat_St_Fn_loss - hat_St_Fd_loss) / params.rhoStR
    hat_Ft = hat_Ft_Fn_gain - hat_Ft_Fn_loss - hat_Ft_Fd_loss
    hat_Fn = - (hat_Fn_proli_loss + hat_Fn_basal_loss) + hat_Fn_angio_gain + hat_Fn_regain
    hat_Fv = hat_Fv_St_gain
    hat_Fd = - hat_Fd_St_loss

    ###################
    rhoSh_Prod = df.Constant(0.0)
    rhoSt_Prod = hat_St
    rhoS_Prod = rhoSh_Prod + rhoSt_Prod
    rhoF_Prod = - rhoS_Prod
    ###################
    nSt_Prod.assign(df.project(hat_St, V1))

    rhoFn_Prod.assign(df.project(hat_Fn, V1))
    rhoFt_Prod.assign(df.project(hat_Ft, V1))
    rhoFv_Prod.assign(df.project(hat_Fv, V1))
    rhoFd_Prod.assign(df.project(hat_Fd, V1))
    ########################################
    v = (u - u_n) / params.dt
    div_v = df.inner(D_S, I)
    dt_nSt = (nSt - nSt_n) / params.dt
    dt_cFn = (1 / params.dt) * (cFn - cFn_n)
    dt_cFt = (1 / params.dt) * (cFt - cFt_n)
    dt_cFv = (1 / params.dt) * (cFv - cFv_n)
    dt_cFd = (1 / params.dt) * (cFd - cFd_n)
    #######################################################
    #Balance of Momentum
    res_mom1 = df.inner(P, df.grad(_u)) * dx
    res_mom2 = - J_S * rhoF_Prod * (params.KF/nF) * df.inner(df.dot(df.grad(p), df.inv(F_S).T), _u) * dx
    bm = res_mom1 + res_mom2
    #Balance of Mass (mixture)
    res_mix1 = J_S * df.inner(D_S, I) * _p * dx
    res_mix2 = - J_S * params.KF * df.inner((df.dot(df.grad(p), df.inv(C_S))), df.grad(_p)) * dx
    res_mix3 = - J_S * ((rhoS_Prod/rhoS) + (rhoF_Prod/params.rhoFR)) * _p * dx
    bfm = res_mix1 + res_mix2 + res_mix3
    #Balance of Volume (Tumor)
    res_bsv1 = J_S * (dt_nSt - nSt_Prod) * _nSt * dx
    res_bsv2 = J_S * nSt * df.inner(D_S, I) * _nSt * dx
    bsv = res_bsv1 + res_bsv2
    #Balance of Concentration (Nutrition)
    res_cFn1 = - J_S * df.dot((- params.DFn * df.dot(df.grad(cFn), df.inv(C_S))
                               - cFn * params.KF * df.dot(df.grad(p), df.inv(C_S))), df.grad(_cFn)) * dx
    res_cFn2 = J_S * (nF * dt_cFn - cFn * (rhoS_Prod/rhoS) - (rhoFn_Prod/params.MFn)) * _cFn * dx
    res_cFn3 = J_S * cFn * df.inner(D_S, I) * _cFn * dx
    bcFn = res_cFn1 + res_cFn2 + res_cFn3
    #Balance of Concentration (Tumor)
    res_cFt1 = - J_S * df.dot((- params.DFt * df.dot(df.grad(cFt), df.inv(C_S))
                               - cFt * params.KF * df.dot(df.grad(p), df.inv(C_S))), df.grad(_cFt)) * dx
    res_cFt2 = J_S * (nF * dt_cFt - cFt * (rhoS_Prod/rhoS) - (rhoFt_Prod/params.MFt)) * _cFt * dx
    res_cFt3 = J_S * cFt * df.inner(D_S, I) * _cFt * dx
    bcFt = res_cFt1 + res_cFt2 + res_cFt3
    #Balance of Concentration (VEGF)
    res_cFv1 = - J_S * df.dot((- params.DFv * df.dot(df.grad(cFv), df.inv(C_S))
                               - cFv * params.KF * df.dot(df.grad(p), df.inv(C_S))), df.grad(_cFv)) * dx
    res_cFv2 = J_S * (nF * dt_cFv - cFv * (rhoS_Prod/rhoS) - (rhoFv_Prod/params.MFv)) * _cFv * dx
    res_cFv3 = J_S * cFv * df.inner(D_S, I) * _cFv * dx
    bcFv = res_cFv1 + res_cFv2 + res_cFv3
    #Balance of Concentration (Drugs)
    res_cFd1 = - J_S * df.dot((- params.DFd * df.dot(df.grad(cFd), df.inv(C_S))
                               - cFd * params.KF * df.dot(df.grad(p), df.inv(C_S))), df.grad(_cFd)) * dx
    res_cFd2 = J_S * (nF * dt_cFd - cFd * (rhoS_Prod/rhoS) - (rhoFd_Prod/params.MFd)) * _cFd * dx
    res_cFd3 = J_S * cFd * df.inner(D_S, I) * _cFd * dx
    bcFd = res_cFd1 + res_cFd2 + res_cFd3

    res_tot = bm + bfm + bsv + bcFn + bcFt + bcFv + bcFd

    # Define problem solution
    solver = nonlinvarsolver(res_tot, w, bcs, params.solv_nTolRel, params.solv_nTolAbs, params.solv_sType)

    #######################################################
    # Initial Conditions
    #######################################################
    # Initialize solution time
    t = 0.0
    u_init = df.Constant([0.0, 0.0, 0.0])
    nF_init = 1.0 - params.nSh0 - params.nSt0
    cFt_init = df.Expression(("ct0*exp(-a*(pow((x[0]-x_source),2)+pow((x[1]-y_source),2)))"), degree=2, ct0=params.cFt0, a=params.a_t, x_source=params.x_source, y_source=params.y_source)
    u_n = df.interpolate(u_init, V2)
    p_n = df.interpolate(df.Constant(params.p0), V1)
    nSt_n = df.interpolate(df.Constant(params.nSt0), V1)
    cFn_n = df.interpolate(df.Constant(params.cFn0), V1)
    cFt_n = df.interpolate(cFt_init, V1)
    cFv_n = df.interpolate(df.Constant(params.cFv0), V1)
    cFd_n = df.interpolate(df.Constant(params.cFd0), V1)

    #write_field2output(params.output_file, u_n, "u", t)
    #write_field2output(params.output_file, p_n, "p", t)
    #write_field2output(params.output_file, nSt_n, "nSt", t)
    #write_field2output(params.output_file, df.project(params.nSh0, V1), "nSh", t)
    #write_field2output(params.output_file, df.project(nF_init, V1), "nF", t)
    #write_field2output(params.output_file, cFn_n, "c_m^Fn", t)
    #write_field2output(params.output_file, df.project(params.cFn_min_growth, V1), "cFn_min_growth", t)
    #write_field2output(params.output_file, df.project(params.cFN_min_necrosis, V1), "cFn_min_necrosis", t)
    #write_field2output(params.output_file, cFt_n, "c_m^Ft", t)
    #write_field2output(params.output_file, cFv_n, "c_m^Fv", t)
    #write_field2output(params.output_file, cFd_n, "c_m^Fd", t)
    #write_field2output(params.output_file, df.project(df.Constant(0.0), V1), "vM_Stress", t)

    #df.assign(w_n.sub(0), u_n)
    df.assign(w_n.sub(1), p_n)
    df.assign(w_n.sub(2), nSt_n)
    df.assign(w_n.sub(3), cFn_n)
    df.assign(w_n.sub(4), cFt_n)
    df.assign(w_n.sub(5), cFv_n)
    df.assign(w_n.sub(6), cFd_n)

    t = t + params.dt
    #######################################################
    # Time loop
    #######################################################
    while t <= params.T_end:
        # Print current time
        df.info("Time: {}".format(t))
        # Calculate current solution
        n_iter, converged = solver.solve()
        print("Time: {}".format(t), "  ", "Converged in steps: {}".format(n_iter), phase)
        # Output solution
        u, p, nSt, cFn, cFt, cFv, cFd = w.split()
        nSt_begin.assign(df.project(df.conditional(df.gt(nSt, 0.0), 1.0, 0.0), V1))
        angio_begin.assign(df.project(angio_begin + df.conditional(df.ge(cFv, params.cFv_angio), 1.0, 0.0), V1))
        nSt_detection.assign(df.project(df.conditional(df.ge(nSt, params.tumor_detection_value), 1.0, 0.0), V1))
        breakout.assign(df.project(df.conditional(df.ge(nSt, params.nSt_max), 1.0, 0.0), V1))
        cFn_VEGF_touch.assign(df.project(H6, V1))
        if phase == "fluid tumor growth":
            if df.assemble(nSt_begin * dx) > 0.0:
                phase = "solid tumor growth"
        elif phase == "solid tumor growth":
            if status == "undetected":
                if df.assemble(nSt_detection * dx) > 0.0:
                    cFd_infusion = InitialConditions(params)
                    cFd_infusion_field = df.interpolate(cFd_infusion, W)
                    u_inf, p_inf, nSt_inf, cFn_inf, cFt_inf, cFv_inf, cFd_inf = cFd_infusion_field.split()
                    df.assign(w.sub(6), df.project(cFd_inf, V1))
                    status = "discovered"
                    phase = "treatment"
        if df.assemble(cFn_VEGF_touch * dx) > 0.0:
            df.assign(w.sub(5), df.project(df.conditional(df.ge(cFv, 0.9 * params.cFv_init), cFv, H6 * (1.0 - H4) * params.cFv_init), V1))
            if df.assemble(angio_begin * dx) > 0.0:
                phase = "angiogenesis"
                angio_activate.assign(df.project(df.conditional(df.gt(angio_begin, 0.0), 1.0, 0.0), V1))
        df.assign(w.sub(2), df.project(df.conditional(df.gt(nSt, 0.9875 * params.nSt_init), nSt, H3 * params.nSt_init), V1))
        nSt_Prod.assign(df.project(hat_St, V1))
        rhoFn_Prod.assign(df.project(hat_Fn, V1))
        rhoFt_Prod.assign(df.project(hat_Ft, V1))
        rhoFv_Prod.assign(df.project(hat_Fv, V1))
        rhoFd_Prod.assign(df.project(hat_Fd, V1))
        #vM_Stress = calcStress_vonMises(T)
        #write_field2output(params.output_file, u, "u", t)
        #write_field2output(params.output_file, p, "p", t)
        write_field2output(params.output_file, nSt, "nSt", t)
        #write_field2output(params.output_file, df.project(nSh, V1), "nSh", t)
        #write_field2output(params.output_file, df.project(nF, V1), "nF", t)
        #write_field2output(params.output_file, cFn, "c_m^Fn", t)
        write_field2output(params.output_file, cFt, "c_m^Ft", t)
        #write_field2output(params.output_file, df.project(df.conditional(df.ge(cFv, 0.0), cFv, 0.0), V1), "c_m^Fv", t)
        write_field2output(params.output_file, cFd, "c_m^Fd", t)
        #write_field2output(params.output_file, df.project(vM_Stress, V1), "vM_Stress", t)
        #write_field2output(output_file, df.project(T(u, p, nS, lambdaS, muS), V3), "Stress", t)
        #write_field2output(output_file, df.project(wF, V2), "w_F", t)
        # Update history fields
        w_n.assign(w)
        # Increment solution time
        t = t + params.dt

    return df.assemble(cFt * J_S * dx)

#########################################################################################
# Simulation
#########################################################################################
params = Parameter()
# Geometry
params.file = "workingdata/Mesh.geo"
params.output_file = df.XDMFFile("solution/29_GrowthExperiment_NewDrugInfusion.xdmf")  # Initializes xdmf file of given name
params.centerparameter = 0.001
params.meshsize = 0.1

# Time discretisation
params.T_end = 120
params.dt = 10

# Source geometry
params.x_source = 0
params.y_source = 0

params.x_infusion = 0
params.y_infusion = 0
params.radius_infusion = 0.06

# Solution algorithm
params.solv_sType = "lu" #"umfpack" #"lu"
params.solv_nTolRel = 1E-7
params.solv_nTolAbs = 1E-8

###################################################
# Material parameters

# Material (healthy Solid)
params.lambdaSh = 3312.0
params.muSh = 662
params.rhoShR = 1190.0
params.nSh0 = 0.4

# Material (tumorous Solid)
params.lambdaSt = 3312.0   # 1.5E7 #
params.muSt = 662  # 1E7 #
params.rhoStR = 1190.0
params.nSt0 = 0.0
params.nSt_max = 0.1   # 1.0 - nSh0
params.nSt_init = 8E-7
params.tumor_detection_value = 9E-5
params.v_St_necrosis = 1E-5 * 86400
params.v_St_growth = 0.35856

params.nS0 = params.nSh0 + params.nSt0

# Material (interstitial Fluid)
params.rhoFR = 993.3
params.KF = 5E-13

#b = df.Constant([0.0, 0.0])
#muS = 500
#lambdaS = 3000
params.p0 = 0.0

###################
# Concentration parameters

# nutritions
params.MFn = 0.18
params.DFn = 6.6E-10 * 86400
params.cFn0 = 1.0
params.cFn_min_growth = 0.35
params.cFn_min_VEGF_prod = 0.36
params.cFN_min_necrosis = 0.9 * params.cFn_min_growth
params.cFn_angio = 0.55
params.Kgr = 0.156
params.v_In_basal = 8.64E-17 #1E-16 #
params.f_proli = 0.864
params.v_Fn_regrowth = 0.0864 * 4

        # tumorous
params.MFt = 2.018E13
params.DFt = 1.5E-13 * 86400
params.kappa_Ft_proliferation = 0.0864
params.cFt_threshold = 9.828212E-13
params.cFt_Ft2nSt = 9.3E-13
params.cFt0 = 1.15E-13
params.a_t = 100     # Steile Gaußglocke
params.NFt = 1E11

        # VEGF
params.MFv = 3.8123E-2 # 50 #
params.DFv = 1.16E-8 * 86400
params.cFv0 = 0.0
params.cFv_init = 1E-13
params.cFv_angio = 2.5E-11
params.cFv_max = 2.5E-9
params.v_Fv_plus = 0.155E-11 * 86400

# drug infusion
params.MFd = 93
params.DFd = 1E-11 * 86400
params.cFd0 = 0.0
params.cFd_infusion_concentration = 1.16773E-7
params.a_d = 800
#cFd_infusion_field = df.Expression(("cFd_infusion_concentration*exp(-a*(pow((x[0]-x_source),2)+pow((x[1]-y_source),2)))"), degree=2, cFd_infusion_concentration=cFd_infusion_concentration , a=a_d, x_source=x_source, y_source=y_source)
params.cFd_min_impact = 5E-9
params.v_Fd_max = 7.88E-6 * 86400
params.reaction_rate_cFd = 3484.51
params.v_Fd = 2.29E-7 * 1.8
params.v_Fd_halflife = 0.1

mesh, facet_function = gen_mesh(params.file, params.centerparameter, params.meshsize, params.output_file)

W, V1, V2 = function_space(mesh)

bc_u1 = df.DirichletBC(W.sub(0).sub(1), df.Constant(0), facet_function, 1)  # uy=0 auf dem unteren Rand
bc_u2 = df.DirichletBC(W.sub(0).sub(0), df.Constant(0), facet_function, 2)  # ux=0 auf linkem Rand

bcs = [bc_u1, bc_u2]

df.set_log_level(30)
mass = solve(W, V1, V2, bcs, params)
print(mass)
