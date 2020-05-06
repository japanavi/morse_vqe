# Morse Potential on a Quantum Computer
This code is designed to run the Variational Quantum Eigensolver (VQE) algorithm from IBM's Quantum Information Software Kit [(QISKit)](https://qiskit.org/documentation/index.html) on Hamiltonians represented as text files. This code was written while participating the in the spring 2020 Science Undergraduate Laboratory Internships [(SULI)](https://science.osti.gov/wdts/suli) program at Brookhaven National Laboratory.

## Directory
* `mathematica_nbs`: Contains all Mathematica notebooks used to create text file representation of Hamiltonians for each basis.
* `out_puts`: Contains all of the outputs generated from running VQE on the Morse potential with 4 qubits in each basis. `_data.pkl` files contain serialized raw data from the simulaiton and can be used to do further analysis on simulation results. `_output.txt` outlines all of the parameters and results in an easily readable table format. 
* `py_files`: Contains two python scripts. `VQE_class.py` is a class that can be used to run VQE, plot the results, and save results. `morse_energy_levels.py` is a script that plots the first four Morse potential Hamiltonians and the first four bound states.
* `tutorials`: Contains a jupyter notebook and necessary Hamiltonian text files for giving an introdcution to the variatonal method and going through a VQE run step by step for the harmonic and anharmonic oscillator.
* `text_file_hamiltonians`: Contains all text file representation Hamiltonians used for VQE. This is so one can try running VQE on their own without having a Mathematica liscense. 

## Before getting started
I recommend creating a conda environment to do all of you work in since the code running without errors is fairly version dependent when using QISKit.

 1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/) (version: `4.8.3`)
 2. Open terminal, run: `conda create -n <environment_name>` (replace `environment_name` with whatever you want to name your environment).
 3. Activate Environment: `conda activate <environment_name>`
 4. Install QISKit: `pip install qiskit==0.16.2`
 5. When finished make sure to deactive the environment: `conda deactivate <environment_name>`
 
### Hamiltonian file name convention
Currently the code gets the state and basis from the file name of the text file representation of the Hamiltonian. The convention is is follows:  `Potential_basis_energystate_sizexsize.txt` Also, after running VQE on many Hamiltonians, we found that the classical optimizers have a hard time when the energy is zero or close to it. A way around this is to hold off on adding a constant  that would result in the energy becoming zero until after the optimizers are done. To impliment this, add `nonzero` anywhere in the file name after the energy state, e.g. `Potential_basis_energystate_nonzero_sizexsize.txt` or `Potential_basis_energystate_sizexsize_nonzero.txt`

* Potential: Name of potential the VQE is being ran on (e.g. Morse or AnharmonicOscillator)
* Basis: How the Hamiltonian is constructed
	* Position <img src="https://render.githubusercontent.com/render/math?math=\rightarrow"> `pos`
	* Oscillator <img src="https://render.githubusercontent.com/render/math?math=\rightarrow"> `osc`
	* Finite Difference <img src="https://render.githubusercontent.com/render/math?math=\rightarrow"> `finite`
* The energy state is representented as <img src="https://render.githubusercontent.com/render/math?math=E_{i}">, for the *i* th energy state
	*  e.g. Ground state <img src="https://render.githubusercontent.com/render/math?math=\rightarrow"> `E0`
	*  First excited state <img src="https://render.githubusercontent.com/render/math?math=\rightarrow"> `E1`
* Size: Size of the Hamiltonian matrix (determined by the number of qubits used)
	*  N qubits <img src="https://render.githubusercontent.com/render/math?math=\rightarrow 2^{N}\times 2^{N}"> matrix

For example, if you were running the Morse potential in the position basis,  calculating the ground state, and used $4$ qubits, the file name would be `Morse_pos_E0_16x16.txt`

## Prerequisites
Make sure you either have the Hamiltonian text files you plan to use in the same directory as this script or input the full path into the `file_path` key in your param dictionary.

If planning to use IBM's Q Experience API: make a python script titled `my_secrets.py` and define a string with your unique IBMQ API token as its value. Ex: `pw = 'your_API_key'`

The main packages used are outlined below. Note that QISKit is very new and the developers release new versions often and a lot of methods and functions get depricated regularly. I will not be updating this code to work with any newer versions of QISKit. I outline the versions of QISKit that work with this code. You can install specific versions of QISKit through pip with the following command: `pip insall qiskit==0.16.2` 

System information: Python 3.8.0 (default, Nov 6 2019, 21:49:08) [GCC 7.3.0]
Conda Version: `4.8.3`
```
os
sys
time
numpy
pickle
pandas
seaborn
datetime
tabulate
matplotlib
```
### QISKit Version Information
| Qiskit Software|Version      |              
|----------------|-------------|
|Qiskit		 |`0.16.2`     |
|Terra           |`0.12.0`     |
|Aer         	 |`0.4.1`      |
|Ignis           |`0.2.0`      |
|Aqua            |`0.6.5`      |
|IBM Q Provider  |`0.5.0`      |

## Example 
Due to the large number of parameters VQE has, I recommed creating a parameter dictionary to organize all of them in an easily readable way. Below is a snippet of code that will run VQE on a Hamiltonian of your choice, save all of the raw data & general output, and generate three plots (energy convergence, best optimizer convergence, and statevectors for each optimizer).
```
params = {
	'file_path':fr'Morse_pos/Morse_pos_E0_16x16.txt',
	'init_state_name':Zero,
	'vf_name':RY,
	'depth':1,
	'shots':1,
	'simulator': 'statevector_simulator'
}

v_q_e = RunVQE(params)					# Runs VQE, prints parameters & results
v_q_e.save_data_to_pkl(f'{v_q_e.file_name}_data.pkl')	#Saves raw data from simulation
v_q_e.save_output(f'{v_q_e.file_name}_output.txt')	#Saves tabular output
v_q_e.plt_energy_convergence()
v_q_e.plt_opt_convergence()
v_q_e.plt_state_vector()
```

## Authors

* **Joshua Apanavicius**

## Acknowledgments

* My mentor for this project, Michael D. McGuigan
* The United States Department of Energy, for funding the SULI program
* Brookhaven National Laboratory, for hosting my stay and allowing me to participate in this project

> Written with [StackEdit](https://stackedit.io/).
