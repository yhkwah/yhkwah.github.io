"""
Computes properties of freely-propagating, premixed, flat flames with
multicomponent transport properties.
A csv input file contains the details of the operating points. Each
operating point is specified by a single row.
"""
from cmath import log
import os as os
import cantera as ct
import csv
import numpy as np
import time as time
# from listtricks import list_of_floats
# ___________________________inputs____________________________________________

st=time.time()
# Path to input csv file
readfrom_path_branch_bs = (r"C:\Users\yihk\Downloads"
                           r"\OneDrive - NTNU\A-cantera\YHK_spreadsheet")

readfrom_path_branch = readfrom_path_branch_bs.replace(os.sep, '/')
#readfrom_csv_filename = 'YHK_ops_part1.csv'
#readfrom_csv_filename = 'test_ops.csv'
readfrom_csv_filename = 'YHK_ops.csv'
#readfrom_csv_filename = 'YHK_ops_pure.csv'


# Name of directory to save results in
saveto_path_high_bs = (r"C:\Users\yihk\Downloads"
                       r"\OneDrive - NTNU\A-cantera\YHK_spreadsheet\results")


saveto_path_high = saveto_path_high_bs.replace(os.sep, '/')
#csv_filename = readfrom_csv_filename.replace('.csv', '') + '_results_part1.csv'
#csv_filename = readfrom_csv_filename.replace('.csv', '') + '_results_test.csv'
#csv_filename = readfrom_csv_filename.replace('.csv', '') + '_results.csv'
csv_filename = readfrom_csv_filename.replace('.csv', '') + '_results_refinedgrid.csv'
#csv_filename = readfrom_csv_filename.replace('.csv', '') + '_results_pure.csv'
#csv_filename = readfrom_csv_filename.replace('.csv', '') + '_results_pure_refinedgrid.csv'

# Simulation parameters6
Tin = 293.0    # unburned gas temperature [K]
P = 101325.0   # pressure [Pa]

# Effective Lewis Number Calculation Inputs
compute_Le_eff = True
#evaluation of Ea_over_R requires calculation of rho_u*SL at different Tb.
#how to do it? estimated by small variations of T_u 
#(Beeckmann et al., 2017) and (Bradley et al., 1998)
#T_u_difference = [-5, 5]  # Kelvin
T_u_difference = [-50, 50]  # Kelvin

# Inputs for solving premixed flame simulation
width = 0.03  # m
loglevel = 1  # amount of diagnostic output (0 to 8)

# Inputs for parsing csv input file
equivalence_ratio_col_header = 'phi'
nitrogen_pct_col_header = 'N2'
kinetic_mechanism_col_header = 'mechanism'
fuel_fraction_col_headers = ['ff_CH4', 'ff_C2H4', 'ff_C3H8',
                                'ff_H2', 'ff_NH3']
fuel_massfraction_col_headers = ['fmf_CH4', 'fmf_C2H4', 'fmf_C3H8',
                                'fmf_H2', 'fmf_NH3']
corresponding_fuel_names = ['CH4', 'C2H4', 'C3H8', 'H2', 'NH3']

# _______________________end_inputs__________________________________


# Make directories
dir_path = saveto_path_high
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    os.makedirs(dir_path + '/detailed')
    print("Directory ", dir_path,  " Created ")
else:
    print("Directory ", dir_path,  " already exists")
    if not os.path.exists(dir_path + '/detailed'):
        os.makedirs(dir_path + '/detailed')

#temperature depending on whether we compute Le_eff
if compute_Le_eff:
    T_u = [Tin + T_u_difference[0], Tin + T_u_difference[1], Tin]
else:
    T_u = [Tin]

# Read operating point details from csv file
readfrom_path_full = (readfrom_path_branch + '/' + readfrom_csv_filename)
op_csvfile = open(readfrom_path_full, newline='')
filereader = csv.reader(op_csvfile, delimiter=',')
# csvfile contents into a list
csvfile_contents = []
for row in filereader:
    # row_float = [float(item) for item in row]
    csvfile_contents.append(row)
op_csvfile.close
csvfile_headers = csvfile_contents[0]
csvfile_data_string = np.array(csvfile_contents[1:])

# Parse input csv file
#find the idx (col no) of phi, N2, mechanism
phi_col = csvfile_headers.index(equivalence_ratio_col_header)
N2_dil_col = csvfile_headers.index(nitrogen_pct_col_header)
mech_col = csvfile_headers.index(kinetic_mechanism_col_header)
#1st col is idx zero!
case_name_array = csvfile_data_string[:, 0]

# Open csv file for writing results and write table headers
csv_file = open(dir_path + '/' + csv_filename, 'w', newline='',
                encoding='utf-8')
writer = csv.writer(csv_file)
diffusivity_header_strings = []

for dh_fuel in corresponding_fuel_names:
    diffusivity_header_strings.append('D_' + dh_fuel + ' (m^2/s)')
diffusivity_header_strings.append('D_O2 (m^2/s)')
#writing all the relevant flame/gas properties, then the diffusivity terms
writer.writerow(csvfile_headers
                + ['S_L0 (m/s)', 'dil_ratio', 'S_L0*dil_ratio',
                   'rho_b (kg/m^3)', 'rho_u (kg/m^3)', 'Tad (K)', 'Ze',
                   'c_p (J/kg/K)', 'k (W/m/K)', 'visc (Pa-s)', 
                   'Le_O2', 'Le_F','Le_F_law','Le_eff','Le_eff_law',
                   'alpha (m^2/s)','thickness [m]'])
                #+ diffusivity_header_strings)

#writer.writerow(csvfile_headers
#                + ['S_L0 (m/s)', 'dil_ratio', 'S_L0*dil_ratio',
#                   'rho_b (kg/m^3)', 'rho_u (kg/m^3)', 'Tad (K)', 'Ze', 'E_a',
#                   'c_p [J/kg/K]', 'k (W/m/K)', 'visc (Pa-s)']
#                + diffusivity_header_strings)
############## writing of table headers ends here

#case_name_array contains all the cases. 
#Use enumerate() to get a counter in a loop

#actual calcul start here, after all the preamble-ish.
for row, case_name in enumerate(case_name_array):
    fuels = []
    fuel_fractions = []
    fuel_massfractions = []
    #looping through all the fuel col of each row
    for fh_index, fuel_header in enumerate(fuel_fraction_col_headers):
        fuel_col = csvfile_headers.index(fuel_header)
        if csvfile_data_string[row, fuel_col] != '0':
            fuels.append(corresponding_fuel_names[fh_index])
            fuel_fractions.append(float(csvfile_data_string[row, fuel_col]))
            
    for fh_index, fuel_header in enumerate(fuel_massfraction_col_headers):
        fuelmass_col = csvfile_headers.index(fuel_header)
        if csvfile_data_string[row, fuelmass_col] != '0':
            #fuels.append(corresponding_fuel_names[fh_index])
            fuel_massfractions.append(float(csvfile_data_string[row, fuelmass_col]))
    # use tuple to store multiple items in a single variable.
    fuels = tuple(fuels)
    phi = float(csvfile_data_string[row, phi_col])
    # N2 dilution as a molar fraction of the fuel-air-nitrogen mixture
    N2_dil = float(csvfile_data_string[row, N2_dil_col])/100
    chemical_mechanism = csvfile_data_string[row, mech_col]
    chemical_mechanism = str(chemical_mechanism); #make this into str
    if chemical_mechanism == 'GRImech30_cerfacs.cti':
        phase_name = 'gri30_multi'
    else:
        phase_name = ''

    if not phase_name:
        gas = ct.Solution(chemical_mechanism)
        gas_u = ct.Solution(chemical_mechanism)
    else:
        gas = ct.Solution(chemical_mechanism, name=phase_name)
        gas_u = ct.Solution(chemical_mechanism, name=phase_name)

    # Construct string of cs fuels for input to set_equivalence_ratio
    # reporting fuel_species in fuel fraction
    fuel_species = ''
    for ss in range(len(fuels)):
        fuel_species = (fuel_species + fuels[ss] + ':'
                        + "%5.3f" % fuel_fractions[ss] + ', ')
    fuel_species = fuel_species[0:-2] #remove the last comma n spacing. 

    # Construct file name for detailed results
    file_name_detres = 'ER_{ER:3.2f}'.format(ER=phi)
    for index in range(len(fuels)):
        file_name_detres = (file_name_detres + fuels[index] + '_'
                            + '{fuel_val:3.2f}')
        file_name_detres = (file_name_detres.
                            format(fuel_val=fuel_fractions[index]))
    file_name_detres = (file_name_detres
                        + 'N2_{N2val:3.2f}'.format(N2val=N2_dil))
    file_name_detres = file_name_detres.replace('.', '')

#why like this? is it cos len(T_u) if compute_Le_eff
    for counter, Tu in enumerate(T_u):
        # set the gas state excluding additional Nitrogen
        gas.set_equivalence_ratio(phi, fuel_species, 'O2:1.0, N2:3.76')

        # dilute the gas with additional Nitrogen
        X = gas.X
        N2_index = gas.species_index('N2')
        X[N2_index] = X[N2_index] + N2_dil/(1-N2_dil)
        gas.X = X
        # print(gas.X)

        gas.TP = Tu, P

        if counter == 0:
            # Make a copy of the unburned gas object
            gas_u.X = X
            gas_u.TP = Tin, P

        # Set up flame object
        f = ct.FreeFlame(gas, width=width)
        #f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
        f.set_refine_criteria(ratio=2, slope=0.03, curve=0.05)
        # f.show_solution()

        # Solve with mixture-averaged transport model
        f.transport_model = 'Mix'
        f.solve(loglevel=loglevel, auto=True)

        T_string = 'Tu_{Tun:4.1f}'.format(Tun=Tu)
        T_string = T_string.replace('.', '')

        # Solve with the energy equation enabled
        file_path_det = (dir_path + '/detailed' + '/'
                         + file_name_detres
                         + T_string)
        #f.save(file_path_det + '.xml', 'mix','solution with mixture-averaged transport')
        # f.show_solution()
        #print('mixture-averaged flamespeed = {0:7f} m/s'
        #      .format(f.u[0]))

        # Solve with multi-component transport properties
        f.transport_model = 'Multi'
        f.solve(loglevel)  # don't use 'auto' on subsequent solves
        # f.show_solution()
        #  print('multicomponent flamespeed = {0:7f} m/s'.format(f.velocity[0]))
        #f.save(file_path_det + '.xml', 'multi','solution with multicomponent transport')

        # write the velocity, temperature, density, and mole fractions
        # to a CSV file
        #f.write_csv(file_path_det + '.csv', quiet=False)

        # # Store the flame speed and the flametemperature
        # flamespeed[ii,jj] = f.velocity[0]
        # flametemperature[ii,jj] = f.T[-1]

        #formula and equation in (Beeckmann et al., 2017)
        
        if compute_Le_eff:
            if counter == 0:
                ln_of_rhou_sl = [log(gas_u.density_mass*(f.velocity[0])).real]
                inv_Tb = [1/(f.T[-1])]
                Ze=0
            elif counter == 1:
                ln_of_rhou_sl.append(log(gas_u.density_mass*(f.velocity[0])).real)
                inv_Tb.append(1/(f.T[-1]))
                #Ea_over_R = -2*((ln_of_rhou_sl[1]-ln_of_rhou_sl[0])
                #                / (inv_Tb[1]-inv_Tb[0]))
                Ea_over_R = -2*((ln_of_rhou_sl[1]-ln_of_rhou_sl[0])
                                / ((inv_Tb[1]-inv_Tb[0])*(ct.gas_constant / 1000)))
                Ea = Ea_over_R * (ct.gas_constant / 1000)
                Ze=0
                #Ze = Ea_over_R*(gas.T-gas_u.T)/(gas.T)**2
            else:
                Tb=f.T[-1]
                Ze = Ea_over_R*(Tb-Tin)/(Tb)**2
        else:
            Ze = 0

    #print(file_name_detres + ': S_L0 = {0:6.4g} and'
    #      ' T_ad = {1:6.4g}'.format(f.velocity[0], gas.T))

    # List of diffusion coefficients
        if chemical_mechanism == 'GRImech30_cerfacs.cti':
            diff_coeffs = ['']*(len(fuel_fraction_col_headers)+1)
            thermal_conductivity = ''
            viscosity = ''
        else:
            #diff_coeffs = []
        #for fuel in corresponding_fuel_names:
            #diff_coeffs.append(gas_u.mix_diff_coeffs_mass
            #                   [gas.species_index(fuel)])
            thermal_conductivity = gas_u.thermal_conductivity
            viscosity = gas_u.viscosity
            #diff_coeffs.append(gas_u.mix_diff_coeffs_mass
                           #[gas.species_index('O2')])
    

    #compute flame thickness 
    grids=f.flame.grid
    T=f.T

    grad=[0]*(len(grids)-1)
    for grid in range(len(grids)-1):
        grad[grid]=(T[grid+1]-T[grid])/(grids[grid+1]-grids[grid])

    thickness = (max(T) -min(T)) / max(grad)

        #write a new Lewis computation here?
        #idea is to compute fuel diffusivity and oxidiser diffusivity
        #mass diffusivity should not be the average diffusivity of all species. 
        #Instead, the Le of the fuel is based on the diffusivity of the fuel in the mixture, 
        #and the Le of the oxidizer is based on the diffusivity of O2 in the mixture. 
    if compute_Le_eff:
        thermal_diff=gas_u.thermal_conductivity/gas_u.cp_mass/gas_u.density_mass
        Le_O2 = (thermal_diff/gas_u.mix_diff_coeffs_mole[gas_u.species_index('O2')])
        D_O2=gas_u.mix_diff_coeffs_mole[gas_u.species_index('O2')];
        Le_F = 0
        Le_F_law = 0
        D_F = 0
        q=0
        #for computing Le_eff by vol (so by mol frac)
        for fh_index, fuel_header in enumerate(fuel_fraction_col_headers):
            fuel_col = csvfile_headers.index(fuel_header)
            if csvfile_data_string[row, fuel_col] != '0':
            #fuel_fractions*(thermal_diff-gas_u.mix_diff_coeffs_mol)
                Lei=(thermal_diff/gas_u.mix_diff_coeffs_mole[gas_u.species_index(corresponding_fuel_names[fh_index])])
                Le_F+=(float(csvfile_data_string[row, fuel_col]))*Lei
                
        #for computing Le_eff by heat release (CK Law) - mass frac?        
        for fh_index, fuel_header in enumerate(fuel_massfraction_col_headers):
            fuelmass_col = csvfile_headers.index(fuel_header)
            if csvfile_data_string[row, fuelmass_col] != '0':       
                #get the mass frac out. how? 
                
            #corresponding_fuel_names[fh_index] - this gives us the fuel name say 'NH3'
            #qn: should this be mix_diff_coeffs_mass or mix_diff_coeffs_mole when using heat release approach?
                Lemi=(thermal_diff/gas_u.mix_diff_coeffs_mass[gas_u.species_index(corresponding_fuel_names[fh_index])])
                #over here, should we be using fuel mass frac instead? 
                qi=(float(csvfile_data_string[row, fuelmass_col]))*gas_u.delta_enthalpy[gas_u.species_index(corresponding_fuel_names[fh_index])]/(Tin*gas_u.cp_mole)
                Le_F_law+=qi*(Lemi-1)
                q+=qi
           #gas.delta_enthalpy - this gives the change in enthalpy for each reaction (in J/kmol)
           #delta_standard_enthalpy - this gives the change in standard-state enthalpy for each reaction (in J/kmol)     
                #qn: is it gas_u or gas for enthalpy? gas_u? since is the unburnt gas diffusing into it? 
        
        #for computing Le_eff by vol (so by mol frac)
        if phi <1:
            PHI=1/phi
            Le_E=Le_O2 #excess reactant (oxidiser for lean)
            Le_D=Le_F #deficient reactant (fuel for lean)
        else:
            PHI=phi
            Le_E=Le_F
            Le_D=Le_O2
            
        #A is a measure of the mixture's strength
        A = 1 + (Ze)*(PHI-1)
        Le_eff=1+((Le_E - 1) + (Le_D - 1) * A) / (1 + A)
        
        #for computing Le_eff by heat release (CK Law) - mass frac?
        Le_F_law=1+Le_F_law/q    
        if phi <1:
            PHI=1/phi
            Le_E=Le_O2 #excess reactant (oxidiser for lean)
            Le_D=Le_F_law #deficient reactant (fuel for lean)
        else:
            PHI=phi
            Le_E=Le_F_law
            Le_D=Le_O2
        A = 1 + (Ze)*(PHI-1)
        Le_eff_law=1+((Le_E - 1) + (Le_D - 1) * A) / (1 + A)
        # Above equation can be rearranged to (Le_E + A*Le_D ) / (1 + A).


    writer.writerow(csvfile_data_string[row].tolist()
                    + [f.velocity[0], gas_u.density_mass/gas.density_mass,
                       f.velocity[0]*gas_u.density_mass/gas.density_mass,
                       gas.density_mass, gas_u.density_mass, gas.T, Ze,
                       gas_u.cp_mass, thermal_conductivity, viscosity,
                       Le_O2, Le_F, Le_F_law, Le_eff, Le_eff_law,thermal_diff, thickness])
                   # + diff_coeffs
csv_file.close

et=time.time()

# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


