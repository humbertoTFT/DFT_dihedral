"""
FFT DIHEDRAL FITTING PROGRAM ---

This script implements the methodology describe in:

Flore-Trujillo, H., T.; Rodríguez-Segura, G., L.; Amador, C.; Dominguez, L.;
Fast Fourier Transform Enables Automated Parametrization of Complex Dihedral 
Potentials in All-Atom and Coarse-Grained Force Fields.

Journal of Chemical Information and Modeling (submited).

Please cite this work if you use this code

"""

import sys
import os
import shutil
from numpy.linalg import norm, solve
from numpy import fft, real, imag, sqrt, array, pi, dot, arccos, arctan2, cross, histogram, mean, std, cos, sin, arctan, isnan, polyfit, linspace,log, argmin, zeros, linalg, radians, argsort, degrees
import matplotlib.pyplot as plt

print("\nDIHEDRAL FIT WITH FOURIER ANALYSIS\n")

lambda_f = 0.5

CONV = 2625.49964  # kJ/mol

plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams.update({'font.size':28})

# PARSING COMMAND LINE ARGUMENTS //////////////////////////////////////////////////////////////////

class Option:
    def __init__(self, func=str, num=1, default=None, description=""):
        self.func = func
        self.num = num
        self.value = default
        self.description = description

    def __nonzero__(self):
        if self.func == bool:
            return self.value != None
        return bool(self.value)

    def __str__(self):
        return self.value and str(self.value) or ""

    def setvalue(self, v):
        if len(v) == 1:
            self.value = self.func(v[0])
        else:
            self.value = [self.func(i) for i in v]


# Description
desc = ""

# Index Description
Index = """Index file with the following directives:

"""

# Option list
options = [
    #   option           type         number  default       description

    ("-crd",      Option(str, 1, None, "Input coordinates file: (.gro)")),
    ("-top",      Option(str, 1, None, "Input topology file: (.itp)")),
    ("-name",     Option(str, 1, None, "Folder/files name")),
    ("-file",     Option(str, 1, None, "File with program parameters")),
    ("-maxf",     Option(str, 1, None, "Maximun multiplicity allowed (Default = 6)")),
    ("-iter",     Option(str, 1, None, "Number of iterations (Defautl = 20)")),
    ("-units",    Option(str, 1, None, "[kj (kJ/mol)/kc (kcal/mol)] (Default = kj)")),
    ("-th",       Option(str, 1, None, "R-squared threshold (Default = 0.98)")),
    ("-pini",     Option(str, 1, None, "Minimum number of allowed frequencies to be tested (Default = 1)")),
    ("-pend",     Option(str, 1, None, "Maximum number of allowed frequencies to be tested (Default = 6)")),
]

# Parsing arguments
args = sys.argv[1:]
if '-h' in args or '--help' in args:
    print("\n", __file__)
    print(desc or "\nUsage for this script:\n")
    for thing in options:
        print(type(thing) != str and "%10s  %s" % (thing[0], thing[1].description) or thing)
    print()
    sys.exit()

# Convert the option list to a dictionary, discarding all comments
options = dict([i for i in options if not type(i) == str])

# Process the command line - list the options that were given
opts = []
while args:
    opts.append(args.pop(0))
    options[opts[-1]].setvalue([args.pop(0) for i in range(options[opts[-1]].num)])


# PROGRAM //////////////////////////////////////////////////////////////////////////////////////
CRD = options["-crd"].value
TOP = options["-top"].value

# Maximum frequency allowed
if options["-maxf"].value == None:
    MAXF = 6
else:
    MAXF = int(options["-maxf"].value)

if options["-th"].value == None:
    THRESHOLD = 0.98
else:
    THRESHOLD = float(options["-th"].value)

if options["-pmin"].value == None:
    MIN_TOP_FREQS = 3
else:
    MIN_TOP_FREQS = int(options["-pmin"].value)

if options["-pmax"].value == None:
    MAX_TOP_FREQS = 6
else:
    MAX_TOP_FREQS = int(options["-pmax"].value)


class molecule:

    def readGRO(INGRO):
        ATOMTYPES = ["H","He","Li","Be","B","C","N","O","F","Ne","Si","P","S","Cl","Ar","Br"]
        gro_atoms = []
        gro_cords = []
        gro_atomtypes = []
        with open(INGRO) as GRO:
            DATA = GRO.readlines()
            gro_resn = DATA[2][5:10]
            for gro_line in DATA[2:-1]:
                gro_atom = gro_line[11:15]

                GROATOM = ""
                for l in gro_atom:
                    if l not in ["0","1","2","3","4","5","6","7","8","9"] and l != " ":
                        GROATOM += l


                # Convert to angstroms
                gro_xcrd = float(gro_line[21:29])*10.0
                gro_ycrd = float(gro_line[29:37])*10.0
                gro_zcrd = float(gro_line[37:45])*10.0

                for at in ATOMTYPES:
                    if at == GROATOM:
                        gro_atomtypes.append(at)

                gro_atoms.append(gro_atom)
                gro_cords.append([gro_xcrd,gro_ycrd,gro_zcrd])

        return gro_atoms,gro_cords,len(gro_atoms),gro_atomtypes,gro_resn


    def readTOP(INTOP):

        with open(INTOP) as TOP:
            DATA = TOP.readlines()
            check_moltype = 0
            n_line = 0
            top_resname = 1
            for line in DATA:
                if "[ moleculetype ]" in line:
                    init_moltype = n_line
                    check_moltype = 1
                elif "[" in line and check_moltype == 1:
                    end_moltype = n_line
                    break
                n_line += 1
            for line in DATA[init_moltype:end_moltype]:
                if not line.isspace() and not ";" in line:
                    top_resname = line.split()[0]

            LINES = []
            for line in DATA:
                LINES.append(line)

        return top_resname,LINES


    n_atoms = readGRO(CRD)[2]
    resname = readGRO(CRD)[4]
    atoms = readGRO(CRD)[0]
    coords = readGRO(CRD)[1]
    atomtypes = readGRO(CRD)[3]
    top_resname = readTOP(TOP)[0]
    top_lines = readTOP(TOP)[1]


INFO = options["-file"].value

class dihedral:

    def readINFO(INFO):

        with open(INFO) as IF:
            info_lines = IF.readlines()
            j_bonders = []
            k_bonders = []
            for iline in info_lines:
                if "axis" in iline:
                    try:
                        dihe_j = int(iline.split("=")[1].split()[0])
                        dihe_k = int(iline.split("=")[1].split()[1])
                    except:
                        raise IndexError("Missign index for dihedral")
                elif "j_bonders" in iline:
                    try:
                        j_bonds = iline.split("=")[1].split()
                        for idx in j_bonds:
                            if idx.isdigit():
                                j_bonders.append(int(idx))
                    except:
                        raise IndexError("Missing index for dihedral")
                elif "k_bonders" in iline:
                    try:
                        k_bonds = iline.split("=")[1].split()
                        for idx in k_bonds:
                            if idx.isdigit():
                                k_bonders.append(int(idx))
                    except:
                        raise IndexError("Missing index for dihedral")

                elif "rotation" in iline:
                    try:
                        rotation = float(iline.split("=")[1])
                    except:
                        raise IndexError("No specifed rotation angle")
                elif "charge" in iline:
                    charge = float(iline.split("=")[1])
                elif "multiplicity" in iline:
                    multiplicity = float(iline.split("=")[1])
                elif "method" in iline:
                    method = str(iline.split("=")[1]).strip("\n")
                elif "basis" in iline:
                    try:
                        basis = str(iline.split("=")[1]).strip("\n")
                    except:
                        basis = " "
                elif "nproc" in iline:
                    nproc = int(iline.split("=")[1])
                elif "memory" in iline:
                    memory = int(iline.split("=")[1])

        n_steps = int((360/rotation) - 1)

        return [dihe_j,dihe_k],j_bonders,k_bonders,n_steps,rotation,charge,multiplicity,method,basis,nproc,memory

    dihe_info = readINFO(INFO)
    dihe_axis = dihe_info[0]
    dihe_jbonds = dihe_info[1]
    dihe_kbonds = dihe_info[2]
    n_steps = dihe_info[3]
    rotation = dihe_info[4]
    charge = dihe_info[5]
    multiplicity = dihe_info[6]
    method = dihe_info[7]
    basis = dihe_info[8]
    n_proc = dihe_info[9]
    memory = dihe_info[10]


class mm_parameters:

    def mdp_params(INFO):
        with open(INFO) as INF:
            info_lines = INF.readlines()
            ff = 0
            lj_fudge = 0.4
            qq_fudge = 0.4
            min_steps = 100
            for iline in info_lines:
                if "ff" in iline:
                    try:
                        force_field = str(iline.split("=")[1]).strip("\n")
                    except:
                        raise NameError("Force field not specified")
                elif "lj_fudge" in iline:
                    try:
                        lj_fudge = float(iline.split("=")[1])
                    except:
                        raise NameError("No specifed lj_fudge")
                elif "qq_fudge" in iline:
                    try:
                        qq_fudge = float(iline.split("=")[1])
                    except:
                        raise NameError("No specifed qq_fudge")
                elif "min_run" in iline:
                    try:
                        min_run = str(iline.split("=")[1]).strip("\n")
                    except:
                        raise NameError("Force field not specified")
                elif "min_nst" in iline:
                    try:
                        min_steps = int(iline.split("=")[1])
                    except:
                        min_steps = 0
                elif "min_dt" in iline:
                    try:
                        min_dt = float(iline.split("=")[1])
                    except:
                        min_dt = 0.01
                elif "min_emtol" in iline:
                    try:
                        min_emtol = float(iline.split("=")[1])
                    except:
                        min_emtol = 250

        return force_field, lj_fudge, qq_fudge,min_run,min_steps,min_dt,min_emtol

    mdp_info = mdp_params(INFO)
    ff = mdp_info[0]
    lj_fudge = mdp_info[1]
    qq_fudge = mdp_info[2]
    min_run = mdp_info[3]
    min_steps = mdp_info[4]
    min_dt = mdp_info[5]
    min_emtol = mdp_info[6]


def gaussianINPUT(name,atomtypes,crds,jobtype,nproc,mem,method,basis,charge,multip,dihe,steps,rotation):
    "Function to prepare gaussian input file .inp"

    INGJF = name + ".inp"
    
    with open(INGJF,"w") as INP:
        INP.write("%nproc={}".format(nproc))
        INP.write("\n%mem={}MB".format(mem))
        INP.write("\n%chk="+name+".chk")
        INP.write("\n\n# {} {} {}\n".format(jobtype,method,basis))
        INP.write("\n{} optimization\n".format(name))
        INP.write("\n{:.0f} {:.0f}\n".format(charge,multip))

        n_atom = 0
        for a in atomtypes:

            atom = a

            # angstroms ---
            xcrd = crds[n_atom][0]
            ycrd = crds[n_atom][1]
            zcrd = crds[n_atom][2] 

            INP.write("{:>3}{:>13.6f}{:>13.6f}{:>13.6f}\n".format(atom,xcrd,ycrd,zcrd))

            n_atom += 1
        INP.write("\n")

        if "modredundant" in jobtype:
            dihe_i = dihe[0]
            dihe_j = dihe[1]
            dihe_k = dihe[2]
            dihe_l = dihe[3]
            INP.write("D {} {} {} {} s {} {:.4f}\n\n".format(dihe_i,dihe_j,dihe_k,dihe_l,steps,rotation))


def readGaussianOutput(GOUT,NATOMS):

    with open(GOUT) as GO:
        GDATA = GO.readlines()

        n_line = 0
        for line in GDATA:

            if "ModRedundant input section has been read" in line:
                line_dihe_data = n_line + 1
            if "Step number" in line and "maximum" in line and "scan point" in line:
                opti_dihe_data = n_line
                break

            n_line += 1

    dihe_i = int(GDATA[line_dihe_data].split()[1])
    dihe_j = int(GDATA[line_dihe_data].split()[2])
    dihe_k = int(GDATA[line_dihe_data].split()[3])
    dihe_l = int(GDATA[line_dihe_data].split()[4])


    dihe_maximum = int(GDATA[opti_dihe_data].split()[8])
    dihe_steps = int(GDATA[opti_dihe_data].split()[-1])

    n_line = 0
    current_step = 0
    DIHE_LINES = []
    for line in GDATA:
        if "Step number" in line and "scan point" in line:
            step_scan = line.split()[-4]

            if step_scan != current_step:
                DIHE_LINES.append(n_line)
                current_step = step_scan

            n_line += 1
    DIHE_LINES.append(n_line)


    ENERGIES = []
    n_data = 1
    n_scf_line = 0
    for line in GDATA:
        if "SCF Done" in line:
            if DIHE_LINES[n_data] - 1 == n_scf_line:
                energy = float(line.split()[4])
                ENERGIES.append(energy)
                n_data += 1
            n_scf_line += 1


    # GEOMETRIES ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    GEOMETRIES = []
    n_data = 1
    n_stdorient_line = 0
    n_line = 0
    DIHE_LINES.append(1000000)   # <<<< FIX LATER
    for line in GDATA:
        if "Standard orientation" in line:
            dihedral_line = DIHE_LINES[n_data] - 1
            if dihedral_line == n_stdorient_line:
                MOLECULE = []
                for atom_line in range(n_line + 5,n_line + NATOMS + 5):
                    atom_data = GDATA[atom_line].split()
                    atom = int(atom_data[1])
                    xcrd = float(atom_data[-3])
                    ycrd = float(atom_data[-2])
                    zcrd = float(atom_data[-1])
                    MOLECULE.append([atom,xcrd,ycrd,zcrd])

                GEOMETRIES.append(MOLECULE)

                n_data += 1
            n_stdorient_line += 1
        n_line += 1


    # Dihedral angles -------------------------------------------
    ANGLES = []
    DIHEDRAL = "D({},{},{},{})".format(dihe_i,dihe_j,dihe_k,dihe_l)
    for line_angle in GDATA:
        if "!" in line_angle and DIHEDRAL in line_angle and not "Scan" in line_angle:
            angle = float(line_angle.split()[3])
            ANGLES.append(angle)


    # Search for positive angle closer to zero :::
    idx_min = 0
    min_ang = abs(ANGLES[0])
    for ang in range(len(ANGLES)):
        angle = abs(ANGLES[ang])
        if angle < min_ang:
            idx_min = ang 
            min_ang = angle


    N = len(ANGLES)
    ANGLES2 = []
    ENERGIES2 = []
    GEOMETRIES2 = []
    for ax in range(idx_min,N):
        ANGLE = ANGLES[ax]
        ANGLES2.append(ANGLE)
        ENERGIES2.append(ENERGIES[ax])
        GEOMETRIES2.append(GEOMETRIES[ax])
    for ay in range(0,idx_min):
        ANGLE = ANGLES[ay]
        ANGLES2.append(ANGLE+360.0)
        ENERGIES2.append(ENERGIES[ay])
        GEOMETRIES2.append(GEOMETRIES[ay])


    return ENERGIES2, GEOMETRIES2, ANGLES2

ATOMIX = {1:"H",2:"He",
          3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",
          11:"Na",12:"Mg",13:"Al",14:"Si",15:"Si",16:"S",17:"Cl"}

def genGRO(MOL,GRONAME,ATOMS,RESN):

    NA = len(ATOMS)

    with open(GRONAME,"w") as GRO:
        GRO.write("\n{:>5d}\n".format(NA))
        no_atom = 1
        for atom in MOL:
            name = ATOMS[no_atom-1]

            # coords in nm
            x = atom[1]*0.1
            y = atom[2]*0.1
            z = atom[3]*0.1

            GRO.write("{:>5.0f}{:<5}{:>5}{:>5.0f}{:>8.3f}{:>8.3f}{:>8.3f}\n".format(1,RESN,name,no_atom,x,y,z))
            no_atom += 1
        GRO.write("   5.00000   5.00000   5.00000\n")

def top_amps(amplitudes,N):

    data = array(amplitudes)

    indices = argsort(data)[-N:][::-1]
    values = data[indices]

    return indices,values

def FFT(X_DATA,MAX_FREQ):

    x = array(X_DATA)
    M = len(x)

    X = fft.fft(x)

    A_n = []
    w_n = []

    Nyquist = M//2

    for n in range(Nyquist+1):

        Re = real(X[n])
        Im = imag(X[n])

        A = (2/M) * sqrt(Re**2 + Im**2)
        w =  - arctan2(Im,Re)/pi

        A_n.append(A)
        w_n.append(w)

    cosines = {}
    for frec in range(1,Nyquist+1):  # Exclude freq. n=0

        # Exclude frequencies higher than MAX_FREQ
        if frec <= MAX_FREQ:
            cosine = {frec: [A_n[frec], w_n[frec]]}
            cosines.update(cosine)

    return cosines

def dephase_FFT(coeffs_fft,gamma,M,stepsize):

    cosines = {n:[0,0] for n in coeffs_fft}

    s = -(gamma/180)/(2*pi/M)

    for n in coeffs_fft:
        A_n = coeffs_fft[n][0]
        w_n = coeffs_fft[n][1]
        v_n = w_n - 2*pi*n*s/(M)

        cosines[n][0] = A_n
        cosines[n][1] = v_n

    return cosines

def genTOP(TOPNAME,FF,LJF,QQF,ITPFILE):
    
    if "gromos" in FF:
        print("Gromacs top file")

        # Generate gromacs topology file :::
        with open(TOPNAME,"w") as TOP:
            TOP.write(";Gromacs topology file\n\n")
            TOP.write("[ defaults ]\n")
            TOP.write("1    3    yes   {:.2f}   {:.2f}\n\n".format(LJF,QQF))

            TOP.write("#include <{}>\n".format(ITPFILE))

            TOP.write("\n")
            TOP.write("[ system ]\nMM dihedral scan\n\n")
            TOP.write("[ molecules ]\n{}    1\n".format(MOLECULE.top_resname))

def MM_mdp(FF,MIN_RUN,MIN_STEPS,MIN_DT,MIN_EMTOL):

    if "gromos" in FF:
        print("Gromacs mdp file:")

        # Generate gromacs min file :::
        with open("min_gro.mdp","w") as MIN:
            MIN.write("integrator    = steep\n")
            if "no" in MIN_RUN:
                MIN.write("nsteps        = 0\n")
            elif "yes" in MIN_RUN:
                MIN.write("nsteps        = {:.0f}\n".format(MIN_STEPS))
            MIN.write("emtol         = {:.1f}\n".format(MIN_EMTOL))
            MIN.write("emstep        = {:.4f}\n".format(MIN_DT))
            MIN.write("nstlist       = 1\n")
            MIN.write("cutoff-scheme = Verlet\n")
            MIN.write("ns_type       = grid\n")
            MIN.write("coulombtype   = PME\n")
            MIN.write("rcoulomb      = 1.2\n")
            MIN.write("rvdw          = 1.2\n")
            MIN.write("pbc           = xyz\n")

        # Generate gromacs md file :::
        with open("md_gro.mdp","w") as MD:
            MD.write("integrator     = md\n")
            MD.write("nsteps         = {:.0f}\n".format(10000))
            #MD.write("nstlist       = 1\n")
            MD.write("cutoff-scheme  = Verlet\n")
            MD.write("constraints    = h-bonds\n")
            MD.write("coulombtype    = PME\n")
            MD.write("vdwtype        = cutoff\n")
            MD.write("vdw-modifier   = force-switch\n")
            MD.write("rlist          = 1.0\n")
            MD.write("rcoulomb       = 1.1\n")
            MD.write("rvdw-switch    = 0.9\n")
            MD.write("rvdw           = 1.0\n")
            MD.write("DispCorr       = EnerPres\n")
            MD.write("lincs-iter     = 2\n")
            MD.write("fourierspacing = 0.25\n")

def plot_fourier(data_dict,N):
    X = linspace(0,2*pi,N)
    X2 = linspace(0,360,N)
    Y = []
    for x in X:
        fourier = 0.0
        for f in data_dict:
            amp = data_dict[f][0]
            phi = data_dict[f][1]*pi
            cosine = amp*(1.0 + cos(f*x - phi))
            fourier += cosine
        Y.append(fourier)
    Y = array(Y)
    return(X2,Y)

def plot_fourier_discrete(data_dict,X_ang_vals):
    Y = []
    for x_val in X_ang_vals:
        x = (x_val*pi)/180.0
        fourier = 0.0
        for f in data_dict:
            amp = data_dict[f][0]
            phi = data_dict[f][1]*pi
            cosine = amp*(1.0 + cos(f*x - phi))
            fourier += cosine
        Y.append(fourier)
    Y = array(Y)
    return Y

ATOMS_MASS = {"H":1.008,"He":4.003,"Li":6,"Be":9,"B":10,"C":12.011,"N":14.007,"O":15.999,"F":18.998,"Ne":20.179,
              "Si":28.086,"P":30.974,"S":32.064,"Cl":35.453,"Br":79.909,"I":126.904}

NAME = options["-name"].value

CWD = os.getcwd()
DIRECTORY = NAME+"_dihe"
PATH = CWD+"/"+DIRECTORY

MOLECULE = molecule()

DIHEDRAL= dihedral()
dihe_j = DIHEDRAL.dihe_axis[0]
dihe_k = DIHEDRAL.dihe_axis[1]
dihe_jbonders = DIHEDRAL.dihe_jbonds
dihe_kbonders = DIHEDRAL.dihe_kbonds
dihe_i = dihe_jbonders[0]
dihe_l = dihe_kbonders[0]

if not os.path.exists(PATH):
    os.mkdir(DIRECTORY)
    os.chdir(DIRECTORY)

    print("Folder for dihedral FFT-fit created: {}_dihe".format(NAME))
    print()
    print("Input file for QM optimization generated")

    gaussianINPUT(NAME+"_opt",MOLECULE.atomtypes,MOLECULE.coords,"opt freq",
            DIHEDRAL.n_proc,DIHEDRAL.memory,DIHEDRAL.method,DIHEDRAL.basis,
            DIHEDRAL.charge,DIHEDRAL.multiplicity," ",0,0)

else:
    os.chdir(PATH)

    # Check if optimization output is in directory

    # QM - FOURIER ANALYSIS ///////////////////////////////////////////////////////////////////////
    if os.path.isfile(NAME+"_scan.out") or os.path.isfile(NAME+"_scan.log"):
        print("Performing QM Fourier Analysis :::")

        with open("dihe_fit.log","w") as LOG:
            LOG.write("Dihedral i,j,k,l: {},{},{},{}\n".format(dihe_i,dihe_j,dihe_k,dihe_l))

        if os.path.isfile(NAME+"_scan.out"):
            OUTPUT_DIHE = NAME+"_scan.out"
        else:
            OUTPUT_DIHE = NAME+"_scan.log"

        QM_ENERGIES = array(readGaussianOutput(OUTPUT_DIHE,MOLECULE.n_atoms)[0])*CONV  # Convert to kJ/mol
        MIN_ENER = min(QM_ENERGIES)
        MAX_ENER = max(QM_ENERGIES)
        X_DIHE_DATA = linspace(0,360,len(QM_ENERGIES))

        DIFF_ENERGY = MAX_ENER - MIN_ENER

        # Zeroing Potential Surface --------------------------
        for qe in range(len(QM_ENERGIES)):
            QM_ENERGIES[qe] = QM_ENERGIES[qe] - MIN_ENER


        # SST for R-squared computation -----------------
        mean_QM = mean(QM_ENERGIES)

        SST = 0.0
        for q in QM_ENERGIES:
            SST += (q-mean_QM)**2.0


        # MM DIHEDRAL SCAN ////////////////////////////////////////////////////////////

        OPT_GEOMS = array(readGaussianOutput(OUTPUT_DIHE,MOLECULE.n_atoms)[1])
        ROT_ANGLS = array(readGaussianOutput(OUTPUT_DIHE,MOLECULE.n_atoms)[2])
        QM_ENERGS = array(readGaussianOutput(OUTPUT_DIHE,MOLECULE.n_atoms)[0])
        STD_ANGLS = [ROT_ANGLS[0]+x*DIHEDRAL.rotation for x in range(len(ROT_ANGLS))]


        # MEASURE DIHEDRALS PHASE SHIFT :::::::::::::::::::::::::::::::::
        PATHS = []
        for path1 in dihe_jbonders:
            for path2 in dihe_kbonders:
                PATHS.append([path1,dihe_j,dihe_k,path2])


        PATHS_MASS = []
        MASS = 0
        for path in PATHS:
            mass_path = 0
            for atom_idx in path:
                atom_in_path = MOLECULE.atomtypes[atom_idx-1]
                atom_mass = ATOMS_MASS[atom_in_path]
                mass_path += atom_mass
                MASS += atom_mass
            PATHS_MASS.append(mass_path)

        WEIGHTS = []
        for M in PATHS_MASS:
            WEIGHTS.append(M/MASS)

        N_PATHS = len(PATHS)
        PATHS_PHASE = [0]

        for path  in range(1,N_PATHS):

            GAMMAS = []

            path_i = dihe_i
            path_j = dihe_j
            path_k = dihe_k
            path_l = dihe_l
            path_i2 = PATHS[path][0]
            path_l2 = PATHS[path][3]
            
            rr = 0
            for geom in OPT_GEOMS:

                crds_i = array([geom[path_i-1][1],geom[path_i-1][2],geom[path_i-1][3]])
                crds_j = array([geom[path_j-1][1],geom[path_j-1][2],geom[path_j-1][3]])
                crds_k = array([geom[path_k-1][1],geom[path_k-1][2],geom[path_k-1][3]])
                crds_l = array([geom[path_l-1][1],geom[path_l-1][2],geom[path_l-1][3]])
                crds_i2 = array([geom[path_i2-1][1],geom[path_i2-1][2],geom[path_i2-1][3]])
                crds_l2 = array([geom[path_l2-1][1],geom[path_l2-1][2],geom[path_l2-1][3]])

                v1 = crds_i - crds_j
                v2 = crds_k - crds_j
                v3 = crds_l - crds_k
                v4 = crds_i2 - crds_j
                v5 = crds_l2 - crds_k

                # Dihedral i-j-k-l :::::::::::::::::::::::::::::::::::::::::::::::

                n21 = cross(v2,v1)
                n23 = cross(v3,v2)
                arg_ijkl = dot(n21,n23)/(norm(n21)*norm(n23))
                if arg_ijkl <= -1.0:
                    dihe_ijkl = 180.0
                elif 1.0 <= arg_ijkl:
                    dihe_ijkl = 0.0
                else:
                    dihe_ijkl = arccos(arg_ijkl)*(180.0/pi)
                n12 = cross(v1,v2)
                sign_dihe_ijkl = dot(n12,v3)
                if sign_dihe_ijkl < 0:
                    dihe_ijkl = 360.0 - dihe_ijkl

                # Dihedral i2-j-k-l2 :::::::::::::::::::::::::::::::::::::::::::::::

                n24 = cross(v2,v4)
                n25 = cross(v5,v2)
                arg_i2jkl2 = dot(n24,n25)/(norm(n24)*norm(n25))
                if arg_i2jkl2 <= -1.0:
                    dihe_i2jkl2 = 180.0
                elif 1.0 <= arg_i2jkl2:
                    dihe_i2jkl2 = 0.0
                else:
                    dihe_i2jkl2 = arccos(arg_i2jkl2)*(180.0/pi)
                n42 = cross(v4,v2)
                sign_dihe_i2jkl2 = dot(n42,v5)
                if sign_dihe_i2jkl2 < 0:
                    dihe_i2jkl2 = 360.0 - dihe_i2jkl2

                D_ijkl = radians(dihe_ijkl)
                D_i2jkl2 = radians(dihe_i2jkl2)

                # Angular difference
                gamma = arctan2(sin(D_i2jkl2-D_ijkl),cos(D_i2jkl2-D_ijkl))

                GAMMAS.append(gamma)

                rr += 1

            # Circular mean 
            GAMMAS = array(GAMMAS)
            circ_mean = arctan2(mean(sin(GAMMAS)),mean(cos(GAMMAS)))

            PATHS_PHASE.append(degrees(circ_mean))
        

        with open("dihe_fit.log","a") as LOG:
            for pathh in range(N_PATHS):
                LOG.write("Dephase = {:>8.2f}\n".format(PATHS_PHASE[pathh]))


        shutil.rmtree("MM",ignore_errors=True)
        os.mkdir("MM")
        os.chdir("MM")
        shutil.copyfile("../../"+CRD,"molecule.gro")

        # ITERATION PROCEDURE /////////////////////////////////

        if options["-iter"].value == None:
            NITER = 20
        else:
            NITER = int(options["-iter"].value)


        MFRQS = MAX_TOP_FREQS - MIN_TOP_FREQS + 1

        R2_matrix = zeros((MFRQS,NITER+1))

        n_frequency = 0

        THRESHOLD_REACH = 0
        MAX_ITER_REACH = 0
        BEST_R_squared = 0
        BEST_P = 0
        BEST_ITER = 0

        for frequencies in range(MIN_TOP_FREQS,MAX_TOP_FREQS+1):

            if THRESHOLD_REACH == 1:
 
                break

            n_frequency += 1

            shutil.copyfile("../../"+TOP,"{}-iter{:02d}-00.itp".format(NAME,frequencies))

            MM_mdp(mm_parameters.ff,mm_parameters.min_run,mm_parameters.min_steps,mm_parameters.min_dt,mm_parameters.min_emtol)

            MM_ITERATIONS = []
            POT_ITERATIONS = []

            R_SQUAREDS = []

            DIHE_PROFILE = [0.0 for d in range(DIHEDRAL.n_steps + 1)]

            N_PARAMS = [0]
            MAX_FREQS = [0]

            for iteration in range(NITER+1):

                genTOP("molecule.top",mm_parameters.ff,mm_parameters.lj_fudge,mm_parameters.qq_fudge,"{}-iter{:02d}-{:02d}.itp".format(NAME,frequencies,iteration))
                
                n_rot = 0

                MM_ENERGIES = []

                with open("ROTATION-{}.gro".format(iteration),"w") as RGRO:

                    #RGRO.write("\n{:>5.0f}\n".format(MOLECULE.n_atoms))

                    for r in OPT_GEOMS:
                        ang_rot = int(STD_ANGLS[n_rot])
                        genGRO(r,NAME+"_rot{:03d}.gro".format(ang_rot),MOLECULE.atoms,MOLECULE.resname)
                        genGRO(r,NAME+"_rot{:03d}_pre.gro".format(ang_rot),MOLECULE.atoms,MOLECULE.resname)

                        os.system("gmx editconf -f  {}_rot{:03d}_pre.gro -o {}_rot{:03d}_pre.gro -d 2".format(NAME,ang_rot,NAME,ang_rot))
                        os.system("gmx grompp -f min_gro.mdp -c {}_rot{:03d}_pre.gro -p molecule.top -o {}_rot{:03d}".format(NAME,ang_rot,NAME,ang_rot))
                        os.system("gmx mdrun -v -deffnm {}_rot{:03d}".format(NAME,ang_rot))
                        os.system("echo 0 | gmx trjconv -f {}_rot{:03d}.trr -s {}_rot{:03d}.tpr -o {}_rot{:03d}_post.gro -pbc whole".format(NAME,ang_rot,NAME,ang_rot,NAME,ang_rot))

                        with open("{}_rot{:03d}.log".format(NAME,ang_rot)) as LOG:
                            DATALOG = LOG.readlines()
                            for line in DATALOG:
                                if "Potential Energy" in line and "=" in line:
                                    PotEner = float(line.split("=")[1])
                                    MM_ENERGIES.append(PotEner)

                        MIN_GRO = []

                        with open("{}_rot{:03d}.gro".format(NAME,ang_rot)) as mGRO:
                            min_gro = mGRO.readlines()
                            for line in min_gro[2:-1]:
                                MIN_GRO.append(line)
                            min_boxsize = min_gro[-1]

                        RGRO.write("\n{:>5.0f}\n".format(MOLECULE.n_atoms))
                        for atom in MIN_GRO:
                            RGRO.write(atom)
                        RGRO.write(min_boxsize)

                        n_rot += 1
                        os.system("rm \#* ")


                MM_min = min(MM_ENERGIES)
                MM_max = max(MM_ENERGIES)

                # Zeroing MM Surface ----------------------------------
                for me in range(len(MM_ENERGIES)):
                    MM_ENERGIES[me] = MM_ENERGIES[me] - MM_min


                # OBTAIN DIHEDREAL PROFILE //////////////////////////////////////////////////

                X_axis_offset = STD_ANGLS[0]

                DIHE_PROFILE_CURRENT = []
                SSE = 0.0
                for i in range(DIHEDRAL.n_steps + 1):
                    qm_energy = QM_ENERGIES[i]
                    mm_energy = MM_ENERGIES[i]
                    tors_energy = qm_energy - mm_energy
                    DIHE_PROFILE[i] = DIHE_PROFILE[i] + lambda_f*tors_energy
                    DIHE_PROFILE_CURRENT.append(tors_energy)
                    SSE += (tors_energy)**2.0

                # R-squared ::::::::::::::::::::::::::::::::::::::::::
                R_squared = 1.0 - (SSE/SST)

                if BEST_R_squared  <= R_squared:
                    BEST_R_squared = R_squared
                    BEST_P = frequencies
                    BEST_ITER = iteration


                R2_matrix[n_frequency-1][iteration] = R_squared


                MIN_PROF = min(DIHE_PROFILE)
                MAX_PROF = max(DIHE_PROFILE)
                X_DIHE_DATA = linspace(0,360,len(DIHE_PROFILE))

                DIFF_ENERGY = MAX_PROF - MIN_PROF

                # FFT fit :::::::::::::::::::::::::::::::::::
                fft_profile = FFT(DIHE_PROFILE,MAXF)
                fourier_series = fft_profile

                fourier_series_top = dict()
                if len(fourier_series) <= frequencies:
                    fourier_series_top = fourier_series
                else:
                    AMPS = []
                    FRQS = []
                    PHAS = []
                    for ff in fourier_series:
                        FRQS.append(ff)
                        AMPS.append(fourier_series[ff][0])
                        PHAS.append(fourier_series[ff][1])
                    TOP_AMPS = top_amps(AMPS,frequencies)[0]
                    TOP_AMPS_SORT = TOP_AMPS.sort()
                    for fff in TOP_AMPS:
                        fourier_series_top.update({FRQS[fff]:[AMPS[fff],PHAS[fff]]})

                FREQUENCIES = []
                for nfreq in fourier_series_top:
                    NMAX = nfreq
                    FREQUENCIES.append(nfreq)
                MAX_FREQS.append(NMAX)

                fourier_pot = fourier_series_top

                print("Fourier Coef:",fourier_pot)
                
                with open("../dihe_fit.log","a") as LOG:
                    LOG.write("\nIteration {:02d} {:02d} \n".format(frequencies,iteration))
                    LOG.write("\nFourier coefficients:\n")
                    for c in fourier_pot:
                        amp = fourier_pot[c][0]
                        pha = fourier_pot[c][1]
                        LOG.write("{:02d} {:.4f} {:.4f}\n".format(c,amp,pha))
                

                n_params = len(fourier_pot)
                N_PARAMS.append(n_params)
                

                fourier_plot = plot_fourier(fourier_pot,100)
                X_fplot = fourier_plot[0]
                Y_fplot = fourier_plot[1]

                min_fourier = min(Y_fplot)
                max_fourier = max(Y_fplot)

                diff_mins = MIN_PROF - min_fourier


                DIHE_fitted = plot_fourier_discrete(fourier_pot,ROT_ANGLS)
                QM_fitted = array(DIHE_fitted) + diff_mins + array(MM_ENERGIES)

                plt.scatter(STD_ANGLS,DIHE_PROFILE,label="$V_i+1$",s=100)
                plt.plot(X_fplot,array(Y_fplot)+diff_mins,label="V_tors")
                plt.legend(bbox_to_anchor=(1.05,1.0),loc="upper left")
                plt.xticks([0,60,120,180,240,300,360])
                plt.grid()
                plt.tight_layout()
                plt.savefig("potential-iter{:02d}-{:02d}".format(frequencies,iteration))
                plt.clf()

                for mm in range(len(MM_ENERGIES)):
                    if MM_ENERGIES[mm] > 300.0:
                        MM_ENERGIES[mm] = 0.0

                STD_ANGLS = [ROT_ANGLS[0]+x*DIHEDRAL.rotation for x in range(len(ROT_ANGLS))]


                plt.title("Dihedral profile")
                plt.plot(STD_ANGLS,QM_ENERGIES,label="QM",linewidth=4)
                plt.plot(STD_ANGLS,MM_ENERGIES,label="MM ($R^2 = {:.4f}$)".format(R_squared),linewidth=4)
                plt.plot(STD_ANGLS,DIHE_PROFILE_CURRENT,label="Differ.",color="gray",linewidth=4)
                plt.legend(bbox_to_anchor=(1.05,1.0),loc="upper left")
                plt.xticks([0,60,120,180,240,300,360])
                plt.xlabel("Angle(°)")
                plt.ylabel("Energy (kJ/mol)")
                plt.grid()
                plt.tight_layout()
                plt.savefig("scan_profiles_{:02d}-{:02d}.png".format(frequencies,iteration))
                plt.clf()

                MM_ITERATIONS.append(MM_ENERGIES)
                POT_ITERATIONS.append(array(Y_fplot)+diff_mins)

                # Outpur scans data ----------------------------------------
                with open("scans_data.csv","w") as SCANS:
                    SCANS.write("x,QM,MM,Diff\n")
                    for i in range(len(STD_ANGLS)):
                        angle = STD_ANGLS[i]
                        qm_ener = QM_ENERGIES[i]
                        mm_ener = MM_ENERGIES[i]
                        diff_ener = DIHE_PROFILE[i]
                        SCANS.write("{:.0f},{:.4f},{:.4f},{:.4f}\n".format(angle,qm_ener,mm_ener,diff_ener))


                if THRESHOLD <= R_squared:
                    THRESHOLD_REACH = 1
                    MAX_ITER_REACH = iteration
                    break

                # DEPHASED DIHEDRLAS :::::::::::::::::::::::::::::::::::::::::::::::

                STEPSIZE = DIHEDRAL.rotation

                DEPHASED_POTS = []

                for P in range(1,N_PATHS):
                    fourier_dephased = dephase_FFT(fourier_pot,PATHS_PHASE[P],DIHEDRAL.n_steps-1,STEPSIZE)
                    DEPHASED_POTS.append(fourier_dephased)

                # Write topology with dihedral potential :::::::::::::::::
                with open("{}-iter{:02d}-{:02d}.itp".format(NAME,frequencies,iteration+1),"w") as FITP:
                    for line in molecule.top_lines:
                        FITP.write(line)
                
                #genTOP(mm_parameters.ff,mm_parameters.lj_fudge,mm_parameters.qq_fudge,molecule.top_lines)

                with open("potential.dict","w") as DICT:
                    DICT.write(str(fourier_pot)+"\n")

                G = 1/N_PATHS

                with open("{}-iter{:02d}-{:02d}.itp".format(NAME,frequencies,iteration+1),"a") as DITP:

                    DITP.write("\n; FFT-fitted dihedrals -------\n")
                    DITP.write("\n[ dihedrals ]\n")

                    # DIHEDRAL IJKL ::::::::::::::::::::::::::::::::::
                    for c in fourier_pot:
                        A = fourier_pot[c][0]*G
                        PHI = fourier_pot[c][1]*180.0 % 360
                        DITP.write("   {:>5}{:>5}{:>5}{:>5}   9   {:>4.0f}   {:.4f}   {:.0f}\n".format(dihe_i,dihe_j,dihe_k,dihe_l,PHI,A,c))
                    DITP.write("\n")

                    for DP in range(1,N_PATHS):

                        p_i = PATHS[DP][0]
                        p_j = PATHS[DP][1]
                        p_k = PATHS[DP][2]
                        p_l = PATHS[DP][3]

                        # DIHEDRAL I2JKL2 ::::::::::::::::::::::::::::::::::
                        for c in DEPHASED_POTS[DP-1]:
                            A = DEPHASED_POTS[DP-1][c][0]*G
                            PHI = DEPHASED_POTS[DP-1][c][1]*180.0 % 360
                            DITP.write("   {:>5}{:>5}{:>5}{:>5}   9   {:>4.0f}   {:.4f}   {:.0f}\n".format(p_i,p_j,p_k,p_l,PHI,A,c))
                        DITP.write("\n")


            with open("iterations.csv","w") as ICSV:
                for a in range(len(STD_ANGLS)):
                    ang = STD_ANGLS[a]
                    qm = QM_ENERGIES[a]
                    ICSV.write("{:.4f},{:.4f},".format(ang,qm))
                    for mm_iter in MM_ITERATIONS:
                        mm = mm_iter[a]
                        ICSV.write("{:.4f},".format(mm))
                    ICSV.write("\n")

            with open("potentials.csv","w") as PCSV:
                anGs = linspace(0,360,100)
                for a in range(100):
                    anG = anGs[a]
                    PCSV.write("{:.4f},".format(anG))
                    for pot_iter in POT_ITERATIONS:
                        pot = pot_iter[a]
                        PCSV.write("{:.4f},".format(pot))
                    PCSV.write("\n")



        plt.imshow(R2_matrix[:,1:],cmap="viridis",interpolation="nearest")
        plt.colorbar()
        plt.savefig("R2_plot.png")

        print()
        print("FFT DIHEDRAL FIT ::::::")
        
        if THRESHOLD_REACH == 0:

            print("R-squared threshold not reached")
            print("Best R-squared score:",BEST_R_squared)
            print("Frequencies required:", frequencies)
            print("Writing final topology to:")
            print("{}_dihefit-FFT.itp".format(NAME))
            shutil.copyfile("{}-iter{:02d}-{:02d}.itp".format(NAME,BEST_P,BEST_ITER),"../../{}_dihefit-FFT.itp".format(NAME))
            shutil.copyfile("scan_profiles_{:02d}-{:02d}.png".format(BEST_P,BEST_ITER),"../../{}_dihefit-FFT.png".format(NAME))

        elif THRESHOLD_REACH == 1:

            print("Fiiting procedure reached intended R-squared threshold")
            print("Threshold = {:.4f}".format(THRESHOLD))
            print("Frequencies required:", BEST_P)
            print("Writing final topology to:")
            print("{}_dihefit-FFT.itp".format(NAME))
            shutil.copyfile("{}-iter{:02d}-{:02d}.itp".format(NAME,BEST_P,MAX_ITER_REACH),"../../{}_dihefit-FFT.itp".format(NAME))
            shutil.copyfile("scan_profiles_{:02d}-{:02d}.png".format(BEST_P,MAX_ITER_REACH),"../../{}_dihefit-FFT.png".format(NAME))

    # QM DIHEDRAL SCAN ////////////////////////////////////////////////////////////////
    elif os.path.isfile(NAME+"_opt.out") or os.path.isfile(NAME+"_opt.log"):
        print("Run QM scan file: {}_scan.inp".format(NAME))
        print("Then place the output file in {}_dihe with name {}_opt.(log/out)".format(NAME,NAME))

        if os.path.isfile(NAME+"_opt.out"):
            OUTPUT_OPT = NAME+"_opt.out"
        else:
            OUTPUT_OPT = NAME+"_opt.log"

        # Prepare gaussian input file for dihedral scan
        init_coords = 0
        with open(OUTPUT_OPT) as OUT:
            DATAOUT = OUT.readlines()
            n_oline = 0
            for oline in DATAOUT:
                if "Standard orientation" in oline:
                    init_coords = n_oline
                n_oline += 1


        OPT_COORDS = []

        for oline in range(init_coords+5,init_coords+5+MOLECULE.n_atoms):
            dataline = DATAOUT[oline]

            opt_xcrd = float(dataline.split()[-3])
            opt_ycrd = float(dataline.split()[-2])
            opt_zcrd = float(dataline.split()[-1])

            OPT_COORDS.append([opt_xcrd,opt_ycrd,opt_zcrd])

        MAIN_DIHEDRAL = [DIHEDRAL.dihe_jbonds[0],DIHEDRAL.dihe_axis[0],DIHEDRAL.dihe_axis[1],DIHEDRAL.dihe_kbonds[0]]

        gaussianINPUT(NAME+"_scan",MOLECULE.atomtypes,OPT_COORDS,"opt=modredundant",
                DIHEDRAL.n_proc,DIHEDRAL.memory,DIHEDRAL.method,DIHEDRAL.basis,
                DIHEDRAL.charge,DIHEDRAL.multiplicity,
                MAIN_DIHEDRAL,DIHEDRAL.n_steps,DIHEDRAL.rotation)

    elif not os.path.isfile(NAME+"_opt.out") or not os.path.isfile(NAME+"_opt.log"):

        print("Run QM optimization file: {}_opt.inp".format(NAME))
        print("Then place the output file in {}_dihe with name {}_opt.(log/out)".format(NAME,NAME))
        gaussianINPUT(NAME+"_opt",MOLECULE.atomtypes,MOLECULE.coords,"opt freq",DIHEDRAL.n_proc,DIHEDRAL.memory,DIHEDRAL.method,DIHEDRAL.basis,DIHEDRAL.charge,DIHEDRAL.multiplicity," ",0,0)

