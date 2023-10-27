"""Example script for porous media coupling.

FIXME: check that tangents are correct and if internal tangents are required, i.e.,
       for internal species
"""
import matplotlib.figure
from rich import print
import roadrunner
# -------------------------------------------------------------------------------------
# load model
# -------------------------------------------------------------------------------------
model_path = "spt_liver.xml"

print(model_path)

# loading the model in roadrunner
r = roadrunner.RoadRunner(str(model_path))

# -------------------------------------------------------------------------------------
# selections
# -------------------------------------------------------------------------------------
# define which subset of variables is part of the readouts
selections = [
    # time
    "time",  # [min] model time
    "tsim",  # [min] simulation time (tracked via rate rule)

    # concentrations
    "[S_ext]",  # [mmole/l] substrate concentration in Vext (fluid phase)
    "[S]",  # [mmole/l] substrate concentration in Vli_nofat (solid phase)
    "[P_ext]",  # [mmole/l] product concentration in Vext (fluid phase)
    "[P]",  # [mmole/l] product concentration in Vli_nofat (solid phase)
    "[T]",  # [mmole/l] toxic concentration in Vli_nofat (solid phase)

    # amounts
    "S_ext",  # [mmole] apap amount in Vext (fluid phase)
    "S",  # [mmole] apap amount in Vli_nofat (solid phase)
    "P",  # [mmole] apap amount in Vli_nofat (solid phase)

    # volumes
    "Vext",  # [l] plasma volume (fluid phase)
    #"Vli_fat",  # [l] liver fat volume (fat phase)
    #"Vli_nofat",  # [l] liver no fat volume (solid phase)
    "Vli",  # [l] total liver volume (solid + fat)

    # other readouts
    "necrosis",  # [dimensionless] boolean flat to set cell necrosis in {0, 1}
    "protein",  # [dimensionless] amount of protein catalyzing reaction in [0, 1]
    #"f_fat",  # [dimensionliess] fat fraction

    # rate of change (sources and sinks)
    "S_ext'",  # [mmole/min] rate of change of amounts d S_ext/dt
    "S'",  # [mmole/min] rate of change of amounts d S/dt
    "P_ext'",  # [mmole/min] rate of change of amounts d P_ext/dt
    "P'",  # [mmole/min] rate of change of amounts d P/dt
    "T'",  # [mmole/min] rate of change of amounts d T/dt

    # uec (unscaled elasticity coefficients), tangents
    "uec(SIM, S_ext)",  # [mmole/min/mmole/l] = [l/min] change of rate with metabolites

    # manual tangent (same as above, but via analytical solution)
    # "d_vS_ext__d_S_ext",  # [mmole/min/mmole/l] = [l/min] change of rate with metabolites
]
# set the selection on roadrunner instance
r.timeCourseSelections = selections

# -------------------------------------------------------------------------------------
# set integrator settings
# -------------------------------------------------------------------------------------
# Some integrator settings are required to handle the very small volumes.

# set tolerances for very small FEM volumes
integrator: roadrunner.Integrator = r.integrator
integrator.setValue("absolute_tolerance", 1e-14)
integrator.setValue("relative_tolerance", 1e-14)


# -------------------------------------------------------------------------------------
# save state
# -------------------------------------------------------------------------------------
# saving model state
state_path = f"{model_path}.state"
r.saveState(str(state_path))

# -------------------------------------------------------------------------------------
# array of instance (load state)
# -------------------------------------------------------------------------------------
# creating a vector of 'empty' roadrunner instance and load state
r_vector = [roadrunner.RoadRunner() for k in range(10)]
for rv in r_vector:
    # loading the state in all instances (selections are already set)
    rv.loadState(state_path)

print("r_vec selections:", r_vector[0].timeCourseSelections)


# -------------------------------------------------------------------------------------
# set changes for given simulator
# -------------------------------------------------------------------------------------
# volumes based on FEM point
# Vext is fluid phase of the point, Vcell is the fat + cell phase
# e.g. if you have in your local FEM point
vol_point = 0.125024 #0.1 [liter] volume of FEM point (volume to simulate, i.e. fluid + solid + fat phase)
f_fluid = 0.3  # [dimensionless] fluid fraction
f_fat = 0.0   # [dimensionless] fat fraction
f_solid = 1 - f_fluid - f_fat  # [dimensionless] solid fraction


# These are the changes which have to be applied
changes = {
    # setting volumes
    "Vext": vol_point * f_fluid,  # [liter] fluid phase volume
    #"Vli_fat": vol_point * f_fat,  # [liter] fat phase volume
    "Vli": vol_point * f_solid,  # [liter] solid phase volume

    # protein amount based on position
    # (varied with position of grid point between 0.0 periportal and 1.0 pericentral)
    "protein": 1.0,  # [dimensionless] pericentral point
    # concentrations of apap in fluid phase (local concentration of you fluid phase)
    "[S_ext]": 1.0,  # [mM] = [mmole/liter] apap concentration in fluid phase
}

# making sure model is in state in which was loaded (not needed, but just to be save)
r.resetToOrigin()

# setting all the changes
for key, value in changes.items():
    r.setValue(key, value)

# -------------------------------------------------------------------------------------
# simulate timestep
# -------------------------------------------------------------------------------------

time = 0.0  # [min]
delta_time = 0.1  # [min]
# simulate a step
s = r.simulate(start=time, end=time + delta_time, steps=1)
#print(s)

#print("Reduced Jacobian")
#Jred = r.getReducedJacobian()
#print(Jred)

# -------------------------------------------------------------------------------------
# access results
# -------------------------------------------------------------------------------------
# source and sink terms (rates)
print("d S_ext/dt", s["S_ext'"], "[mmole/min]")
print("d P_ext/dt", s["P_ext'"], "[mmole/min]")

# concentrations (fluid phase)
print("[S_ext]", s["[S_ext]"], "[mmole/l]")
print("[P_ext]", s["[P_ext]"], "[mmole/l]")

# concentrations (solid phase)
print("[S]", s["[S]"], "[mmole/l]")
print("[P]", s["[P]"], "[mmole/l]")
print("[T]", s["[T]"], "[mmole/l]")

# tangents dv/dc = [mmole/min]/[mmole/liter]
# FIXME: check the values
# print("d_vapap_ext__d_apap_ext", s["d_vapap_ext__d_apap_ext"], "[l/mmole]")
# print("uec", s["uec(APAPIM, apap_ext)"], "[l/mmole]")

# necrosis state
print("necrosis", s["necrosis"], "[dimensionless]")


# same simulation with oneStep (not relevant)
# r.resetToOrigin()
# for key, value in changes.items():
#     r.setValue(key, value)
# r.oneStep(currentTime=0.0, stepSize=0.1)
# print(r.getSelectedValues())
#
# print(
#     "unscaled elasticity of APAPIM with respect to apap_ext:",
#     r["uec(APAPIM, apap_ext)"],
# )

# -------------------------------------------------------------------------------------
# Timecourse
# -------------------------------------------------------------------------------------
# full timecourse as reference output; using tsim to track simulation time
# this example shows the difference between the `time` variable and actual simulation time
r.resetToOrigin()
for key, value in changes.items():
    r.setValue(key, value)

tend = 120  # [min]
s1 = r.simulate(start=0.0, end=tend, steps=10) #*60
s2 = r.simulate(start=0.0, end=tend, steps=10) #*60

from matplotlib import pyplot as plt
f: matplotlib.figure.Figure
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
f.subplots_adjust(wspace=0.3)

ax1.set_ylabel("concentration [mmole/l]")
ax1.set_xlabel("tsim [min]")

ax3.set_ylabel("concentration [mmole/l]")
ax3.set_xlabel("time [min]")
ax2.set_ylabel("tsim [min]")
ax2.set_xlabel("time [min]")

for s in [s1, s2]:
    ax1.plot(s['tsim'], s['[S_ext]'], label="[S_ext]")
    ax1.plot(s['tsim'], s['[S]'], label="[S]")
    ax1.plot(s['tsim'], s['[P_ext]'], label="[P_ext]")
    ax1.plot(s['tsim'], s['[P]'], label="[P]")
    ax1.plot(s['tsim'], s['[T]'], label="[T]")

    ax2.plot(s['time'], s['tsim'], label="tsim")

    ax3.plot(s['time'], s['[S_ext]'], label="[S_ext]")
    ax3.plot(s['time'], s['[S]'], label="[S]")
    ax3.plot(s['time'], s['[P_ext]'], label="[P_ext]")
    ax3.plot(s['time'], s['[P]'], label="[P]")
    ax3.plot(s['time'], s['[T]'], label="[T]")

for ax in (ax1, ax2, ax3):
    ax.legend()

plt.show()
f.savefig(f"{model_path}.png", bbox_inches="tight")

from matplotlib import pyplot as plt
f: matplotlib.figure.Figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
f.subplots_adjust(wspace=0.3)

ax1.set_ylabel("concentration [mmole/l]")
ax2.set_ylabel("concentration [mmole/l]")
ax1.set_xlabel("time [min]")
ax2.set_xlabel("time [min]")
for s in [s1]:
    ax1.plot(s['tsim'], s['[S_ext]'], label="[S_ext]")
    ax1.plot(s['tsim'], s['[P_ext]'], label="[P_ext]")


    ax2.plot(s['time'], s['[S]'], label="[S]")
    ax2.plot(s['time'], s['[P]'], label="[P]")
    ax2.plot(s['time'], s['[T]'], label="[T]")

for ax in (ax1, ax2):
    ax.legend()

plt.show()
f.savefig(f"{model_path}2.png", bbox_inches="tight")



import matplotlib.pyplot as plt
import csv
 
X = []
Y = []
 
with open('t_int.txt', 'r') as data:
    #plotting = csv.reader(datafile, delimiter=' ')
     
    for line in data:
        p = line.split()
        X.append(float(p[0]))
        Y.append(float(p[1]))
 

plt.plot(X, Y, color="orange", linewidth=3)
plt.scatter(s['time']*60, s['[T]'])#'g' ,linewidth=3)
#plt.title('coupled vs single roadrunner instance')
#plt.xlabel('time [s]')
#plt.ylabel('Concentration T_int [mM]')
plt.legend(["FEBio/libRoadRunner","libRoadRunner"], loc ="lower right") #Portal Field
plt.show
plt.savefig('T_int.png')
