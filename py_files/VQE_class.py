# Import useful packages
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from tabulate import tabulate
import matplotlib.pyplot as plt

# Import QISKit
from qiskit import Aer, execute, BasicAer
from qiskit.aqua.components.initial_states import Zero
from qiskit.aqua.algorithms.adaptive import VQE
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.variational_forms import RY, RYRZ
from qiskit.aqua.operators.op_converter import to_weighted_pauli_operator
from qiskit.aqua.components.optimizers import COBYLA, SPSA, L_BFGS_B, SLSQP
from qiskit.aqua.operators import (TPBGroupedWeightedPauliOperator,
                                    WeightedPauliOperator,
                                    MatrixOperator)

# For IBMQ
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor, backend_monitor, backend_overview

class RunVQE:
    """Runs VQE via IBM's QISKit

    Attributes:
        file_path (str): File path to text file representation of Hamiltonian
        file_name (str): File name of text file representation of Hamiltonian
        init_state_name (QISKit object): Initial state
        vf_name (QISKit object): Variational Form
        depth (int): Quantum depth for circuit
        shots (int): Number of shots
        n_qubits (int): Number of qubits used
        n_paulis (int): Number of pauli terms generated
        ref (float): Exact energy of Hamiltonian
        optimizers (list of strs): List of optimzers used in VQE run

        The following dictionaries all have the optimzers used in VQE run as keys:
        algos (dict of VQE objects): QISKit VQE objects
        algo_results (dict): Dictionary of algorithm results from each VQE run
        result_df (DataFrame): Contains all convergence values from each
            optimizer used in the run. The index represents the convergence counts

    """
    def __init__(self, kwargs, use_ibmq=False):
        """Constructor method

        Args:
            kwargs (dict): Key word argument dictionary containing parameters
                for VQE run

            save_figs (bool): If True will save figures generated in current
                directory if the plt_state_vector(), plt_opt_convergence(), or
                plt_energy_convergence() methods are called

            use_ibmq (bool): If True will activate IBMQ account and run on an
                IBMQ backend (default is least busy, see get_least_busy_device()
                method below for more details)

        """
        self.file_path = kwargs['file_path']
        self.file_name = self.file_path.split("/")[-1].replace('.txt', '')

        self.init_state_name = kwargs['init_state_name']
        self.vf_name = kwargs['vf_name']
        self.depth = kwargs['depth']
        self.shots = kwargs['shots']
        self.add_k_count = 0

        if use_ibmq:
            self.activate_ibmq()
            self.device = self.get_least_busy_device()
            self.backend = self.provider.get_backend(self.device)
        else:
            self.device = kwargs['simulator']
            self.backend = BasicAer.get_backend(self.device)

        self.run_VQE()
        if 'nonzero' in self.file_name:
            self._add_k(25)

        print(self.result_table())

    def construct_wpo_operator(self):
        """Returns Weighted Pauli Operator (WPO) constructed from Hamiltonian
            text file.

        Args:
            None

        Returns:
            qubit_op (QISKit WPO object): Weighted Pauli Operator
        """
        # Loading matrix representation of Hamiltonian txt file
        H = np.loadtxt(self.file_path)
        # Converting Hamiltonian to Matrix Operator
        qubit_op = MatrixOperator(matrix=H)
        # Converting to Pauli Operator
        qubit_op = to_weighted_pauli_operator(qubit_op)
        self.num_qubits = qubit_op.num_qubits
        self.num_paulis = len(qubit_op.paulis)
        return qubit_op

    def exact_eigensolver(self, qubit_op):
        """Returns exact solution of eigen value problem

        Args:
            qubit_op (object): Weighted Pauli Operator

        Returns:
            exact_energy (float): Exact energy of Hamiltonian
        """
        # Solving system exactly for reference value
        ee = ExactEigensolver(qubit_op)
        result = ee.run()
        exact_energy = result['energy']
        return exact_energy


    def run_VQE(self):
        """Runs the Variational Quantum Eigensolver (VQE)"""
        qubit_op = self.construct_wpo_operator()
        self.ref = self.exact_eigensolver(qubit_op)

        # Setting initial state, variational form, and backend
        init_state = self.init_state_name(self.num_qubits)
        var_form = self.vf_name(self.num_qubits,
                          depth=self.depth,
                          entanglement='linear',
                          initial_state=init_state)


        # Don't use SPSA if using a noiseless simulator
        if self.device == 'statevector_simulator':
            optimizers = [COBYLA, L_BFGS_B, SLSQP]
        else:
            optimizers = [COBYLA, SPSA]

        print(self.param_table())

        # Initializing empty lists & dicts for storage
        df = pd.DataFrame()
        algos = {}
        algo_results = {}
        for optimizer in optimizers:
            # For reproducibility
            aqua_globals.random_seed = 250
            print(f'\rOptimizer: {optimizer.__name__}          ', end='')

            counts = []
            values = []
            params = []
            def store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)
                params.append(parameters)

            # Running VQE
            algo = VQE(qubit_op,
                       var_form,
                       optimizer(),
                       callback=store_intermediate_result)
            quantum_instance = QuantumInstance(backend=self.backend, shots=self.shots)
            algo_result = algo.run(quantum_instance)

            df[optimizer.__name__] = pd.Series(data=values, index=counts)
            algos[optimizer.__name__] = algo
            algo_results[optimizer.__name__] = algo_result

        print('\rOptimization complete')

        self.algos = algos
        self.result_df = df
        self.algo_results = algo_results
        self.optimizers = algo_results.keys()

        self.results_to_save = {
        'file_path': self.file_path,
        'ref': self.ref,
        'n_qubits': self.num_qubits,
        'n_paulis': self.num_paulis,
        'depth': self.depth,
        'n_shots': self.shots,
        'algo_results': self.algo_results,
        'result_df': self.result_df
        }

    def _add_k(self, k):
        """Adds a constant k to energy to get around optimizer error when
            optimizing near zero

        Args:
            k (int or float): constant to add

        Returns:
            None
        """
        if self.add_k_count != 0:
            return 'Method already called, to prevent adding k too many times no operation has been carried out.'
        else:
            self.ref += k
            self.result_df = self.result_df.apply(lambda x: x + k)
            for opt in self.optimizers:
                self.algo_results[opt]['energy'] += 25
            print('\n**WARNING** Outside operation has been performed on algorithm results')

            self.add_k_count += 1

    def save_data_to_pkl(self, out_name):
        """Saves results_to_save dict to pickle file

        Args:
            out_name (str): File name/path for data to be written to

        Returns:
            None
        """
        with open(out_name, 'xb') as file:
            pickle.dump(self.results_to_save, file)
        print(f'Data wiitten to {out_name}')

    def save_output(self, out_name):
        """Saves out put from param_table and result_table to text file

        Args:
            out_name (str): File name/path for data to be written to

        Returns:
            None
        """
        with open(out_name, 'x') as file:
            file.write(self.param_table())
            file.write('\n')
            file.write(self.result_table())
        print(f'Output wiitten to {out_name}')

    def load_data(self, data_file):
        """Loads data from pickle file

        Args:
            data_file (str): pickle file containing data to load

        Returns:
            None
        """
        with open(data_file, 'rb') as file:
            data = pickle.load(file)
        return data

    def find_total_time(self):
        """Finds total simulation time in seconds"""
        eval_times = []
        for opt in self.optimizers:
            eval_times.append(self.algo_results[opt]['eval_time'])
        return round(sum(eval_times))

    def get_state(self):
        """Returns State of Hamiltonian based on state mapping"""
        hamiltonian_info = self.file_name.split('_')
        state_mapping = {
                    'E0':'Ground State',
                    'E1':'1st Excited',
                    'E2':'2nd Excited',
                    'E3':'3rd Excited',
                    'E4':'4th Excited'
        }
        for key in state_mapping:
            if key in hamiltonian_info:
                state = state_mapping[key]
            else:
                pass
        return state

    def get_basis(self):
        """Returns basis of Hamiltonian based on basis mapping"""
        hamiltonian_info = self.file_name.split('_')
        basis_mapping = {
                    'pos': 'Position',
                    'osc': 'Oscillator',
                    'finite': 'Finite Difference'
                    }
        for key in basis_mapping:
            if key in hamiltonian_info:
                basis = basis_mapping[key]
            else:
                pass
        return basis

    def param_table(self):
        """Returns string outlining parameters for current VQE run"""
        run_settings = [['State', self.get_state()],
              ['Basis', self.get_basis()],
              ['Backend', self.device],
              ['InitState', self.init_state_name.__name__],
              ['VarForm', self.vf_name.__name__],
              ['Depth', self.depth],
              ['# Shots', self.shots],
              ['# Qubits', self.num_qubits],
              ['# Paulis', self.num_paulis]
        ]

        out_str = f"""{datetime.utcnow()}\nRunning: {self.file_path}\n{tabulate(run_settings, tablefmt="fancy_grid", stralign='right')}"""
        return out_str

    def result_table(self):
        """Returns string outlining results of VQE simulation for each optimizer"""

        ref_str = f'Reference Value: {self.ref}'
        t_str = f'\n{self.find_total_time()} s to complete'
        vals = []
        for opt in self.optimizers:
            if 'nonzero' in self.file_name:
                vals.append([opt,
                             self.algo_results[opt]['energy'],
                             abs((self.algo_results[opt]['energy'] - self.ref)/self.ref)*100])

        sorted_vals = sorted(vals, key=lambda x: x[2])
        headers = ['Optimizer', 'VQE Energy', '% Error']
        out_str = f"""{t_str}\n{ref_str}\n{tabulate(sorted_vals,tablefmt="fancy_grid",headers=headers,numalign="right")}"""
        return out_str

    def plt_energy_convergence(self, save_figs=True):
        """Graphs optimizer energy convergence and convergence of most
            effective optimizer.
        """
        plt.figure(figsize=(10, 5))
        for opt in self.optimizers:
            self.result_df[opt].apply(lambda x: abs(x - self.ref)).plot(logy=True, label=opt)

        plt.title('Optimizer Energy Convergence', size=24)
        plt.xlabel('Evaluation Count', size=18)
        plt.ylabel('Energy Difference', size=18)
        plt.legend(fontsize='x-large')
        sns.despine()
        if save_figs:
            if os.path.exists(fr'{self.file_name}_energy_convergence.png'):
                plt.savefig(fr'{self.file_name}_energy_convergence(1).png');
            else:
                plt.savefig(fr'{self.file_name}_energy_convergence.png');
        else:
            plt.show()

    def find_most_precise(self):
        """Returns name of most precise optimizer"""
        p_err_mins = {}
        for opt in self.optimizers:
            p_err = self.result_df[opt].apply(lambda x: abs((x - self.ref)/self.ref)*100)
            p_err_mins[opt] = p_err.min()

        most_precise = sorted(p_err_mins.items(), key=lambda x: x[1])[0][0]
        return most_precise

    def plt_opt_convergence(self, save_figs=True):
        """Plots most effective optimizer"""

        most_precise = self.find_most_precise()

        fig, ax = plt.subplots(figsize=(10, 5))
        self.result_df[most_precise].plot(label=f"{most_precise} = {self.algo_results[most_precise]['energy']:.6f}")

        ax.hlines(y=self.ref,
                  xmin=0,
                  xmax=len(self.result_df[most_precise]),
                  colors='r',
                  label=f'Exact = {self.ref:.6f}')

        plt.title('Convergence', size=24)
        plt.xlabel('Optimization Steps', size=18)
        plt.ylabel('Energy', size=18)
        plt.legend(fontsize='x-large')
        sns.despine()
        if save_figs:
            if os.path.exists(fr'{self.file_name}_opt_convergence.png'):
                plt.savefig(fr'{self.file_name}_opt_convergence(1).png');
            else:
                plt.savefig(fr'{self.file_name}_opt_convergence.png');
        else:
            plt.show()

    def plt_state_vector(self, save_figs=True):
        """Plots state vector for each optimizer"""
        plt.figure(figsize=(10, 5))

        for optimizer in self.optimizers:
            circ = self.algos[optimizer].construct_circuit(self.algo_results[optimizer]['opt_params'])

            e0wf = execute(circ[0],
                           Aer.get_backend(self.device),
                           shots=1).result().get_statevector(circ[0])

            plt.plot(np.flip(e0wf.real), label=f'{optimizer}')

        plt.title('State Vectors', size=24)
        plt.legend(fontsize='x-large')
        sns.despine()
        if save_figs:
            if os.path.exists(fr'{self.file_name}_st_vecs.png'):
                plt.savefig(fr'{self.file_name}_st_vecs(1).png');
            else:
                plt.savefig(fr'{self.file_name}_st_vecs.png');
        else:
            plt.show()


    def activate_ibmq(self):
        """Activates IBMQ account and gets IBMQ provider Note: you will get an
            import error if you do not have a python scipt named my_secrets.py
            with your IBMQ API token in the same directory as this file
        """
        from my_secrets import pw
        key = pw

        try:
            IBMQ.enable_account(key)
        except:
            print('IBMQAccountError: Account already activated')
        finally:
            self.provider = IBMQ.get_provider('ibm-q')

    def get_least_busy_device(self):
        """Returns name (str) of least busy IBMQ backend that is not a
            simulator and has more than one qubit.
        """
        lb_backend = least_busy(self.provider.backends(filters=lambda x: not x.configuration().simulator and x.configuration().n_qubits > 1))

        return lb_backend.name()

# Uncomment and run with valid Hamiltonian file
def main():
    params = {
        'file_path':fr'Morse_pos/Morse_pos_E0_16x16.txt',
        'init_state_name':Zero,
        'vf_name':RY,
        'depth':1,
        'shots':1,
        'simulator': 'statevector_simulator'
    }

    v_q_e = RunVQE(params)
    v_q_e.save_data_to_pkl(f'{v_q_e.file_name}_data.pkl')
    v_q_e.save_output(f'{v_q_e.file_name}_output.txt')
    v_q_e.plt_energy_convergence()
    v_q_e.plt_opt_convergence()
    v_q_e.plt_state_vector()


if __name__ == "__main__":
    main()
