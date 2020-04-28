


> Written with [StackEdit](https://stackedit.io/).
# Morse Potential on a Quantum Computer
This code is designed to run the Variational Quantum Eigensolver (VQE) algorithm from IBM's Quantum Information Software Kit [(QISKit)](https://qiskit.org/documentation/index.html) on Hamiltonians represented as text files. This code was written while participating the in the spring 2020 Science Undergraduate Laboratory Internships [(SULI)](https://science.osti.gov/wdts/suli) program at Brookhaven National Laboratory.

## Directory
* `tutorials`: Contains a jupyter notebook and necessary Hamiltonian text files. For giving an introdcution to the variatonal method and going through a VQE run step by step for the harmonic and anharmonic oscillator.
* `vqe`:
* `mathematica_nbs`: Contains all Mathematical files used to generate the text file Hamiltonians for the Morse potential used for VQE in each basis.

## Before getting started
I recommend creating a conda environment to do all of you work in since the code running without errors is fairly version dependent when using QISKit.

 1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/) (version: `4.8.3`)
 2. Open terminal, run: `conda create -n environmentname` (replace `environmentname` with whatever you want to name your environment).
 3. Activate Environment: `conda activate environmentname`
 4. Install QISKit: `pip install qiskit==0.16.2`

### Hamiltonian file name convention
Currently the code gets the state and basis from the file name of the text file representation of the Hamiltonian. The convention is is follows:  `Potential_basis_energystate_sizexsize.txt` Also, after running VQE on many Hamiltonians, we found that the classical optimizers have a hard time when the energy is zero or close to it. A way around this is to hold off on adding a constant  that would result in the energy becoming zero until after the optimizers are done. To impliment this, add `nonzero` anywhere in the file name after the energy state, e.g. `Potential_basis_energystate_nonzero_sizexsize.txt` or ``

* Potential: Name of potential the VQE is being ran on (e.g. Morse or AnharmonicOscillator)
* Basis: How the Hamiltonian is constructed
	* Position $\rightarrow$ `pos`
	* Oscillator $\rightarrow$ `osc`
	* Finite Difference $\rightarrow$ `finite`
* The energy state is representented as $E_{i}$, for the $i$th energy state
	*  e.g. Ground state $\rightarrow$ `E0`
	*  First excited state $\rightarrow$ `E1`
* Size: Size of the Hamiltonian matrix (determined by the number of qubits used)
	*   $N$ qubits $\rightarrow 2^{N}\times 2^{N}$ matrix

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
|Qiskit			     |`0.16.2`     |
|Terra           |`0.12.0`     |
|Aer         	   |`0.4.1`      |
|Ignis           |`0.2.0`      |
|Aqua            |`0.6.5`      |
|IBM Q Provider  |`0.5.0`      |

## Running VQE

 1. Instantiate `RunVQE` class: `v_q_e = RunVQE(params)`
	 * This will automatically print the run's settings, run VQE, and print a results summary.
 2. Save serialized data: `v_q_e.save_data_to_pkl('your_file_name.pkl')`
 3. Save Output (Parameters & Results table): `v_q_e.save_output('your_file_name.txt')`

 List item

## Authors

* **Joshua Apanavicius**

## Acknowledgments

* My mentor for this project, Michael D. McGuigan
* The United States Department of Energy, for funding the SULI program
* Brookhaven National Laboratory, for hosting my stay and allowing me to participate in this project
