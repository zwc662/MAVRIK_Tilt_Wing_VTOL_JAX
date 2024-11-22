import os
import glob
import pickle
import jax
from jax import random
from jax.example_libraries import stax
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import datetime

import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from jax_mavrik.mavrik_types import StateVariables, ControlInputs


current_dir = os.path.dirname(__file__)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class SystemID:
    def __init__(self): 
        self.states = None
        self.controls = None
        self.params = None
        self.predict = None
        self.scaler = StandardScaler()
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(0.001)
        self.opt_state = None

        self.initialize_nn()

    def build_dataset(self):
        dataset = []
        for i in range(len(self.states) - 1):
            current_state = self.states[i]
            current_control = self.controls[i]
            next_state = self.states[i + 1]
            input_vector = jnp.concatenate([current_state, current_control])
            dataset.append((input_vector, next_state))
        return dataset

    def load_data(self):
        data_path = os.path.join(current_dir, 'data')
        pt_files = glob.glob(os.path.join(data_path, '*.pt'), recursive=False)
        data = [pickle.load(open(file, 'rb')) for file in pt_files]

        states = []
        controls = []
        for trajectories in data:
            for trajectory in trajectories:
                if not isinstance(trajectory, dict):
                    continue
                states.extend(trajectory['state'])
                controls.extend(trajectory['control'])

        self.states = jnp.array(states)
        self.controls = jnp.array(controls[:-1])  # Ignore the last control

        # Standardize the data
        self.states_normalized = self.scaler.fit_transform(self.states)
        self.controls_normalized = self.scaler.fit_transform(self.controls)

        # Build the dataset
        dataset = self.build_dataset()
        inputs, targets = zip(*dataset)
        inputs = jnp.array(inputs)
        targets = jnp.array(targets)

        # Split the data into training and testing sets
        self.inputs_train, self.inputs_test, self.targets_train, self.targets_test = train_test_split(
            inputs, targets, test_size=0.2, random_state=42
        )

    def create_nn(self):
        init_random_params, self.predict = stax.serial(
            stax.Dense(128), stax.Relu,
            stax.Dense(128), stax.Relu,
            stax.Dense(len(StateVariables._fields))
        )
        return init_random_params
   
    def initialize_nn(self):
        key = random.PRNGKey(0)
        init_random_params = self.create_nn()
        _, self.params = init_random_params(key, (-1, len(StateVariables._fields) + len(ControlInputs._fields)))
        self.opt_state = self.opt_init(self.params)

    def nn_loss(self, params, inputs, targets):
        preds = self.predict(params, inputs)
        return jnp.mean((preds - targets) ** 2)

    def update(self, params, inputs, targets, lr):
        grads = jax.grad(self.nn_loss)(params, inputs, targets)
        self.opt_state = self.opt_update(0, grads, self.opt_state)
        return self.get_params(self.opt_state)
  
    def train_nn(self, run_name: str = '', epochs=500_000, lr=0.001, test_interval=1000):
        train_losses = []
        test_losses = []
        infos = {k: [] for k in StateVariables._fields}
        for epoch in range(epochs):
            self.params = self.update(self.params, self.inputs_train, self.targets_train, lr)
            if epoch % test_interval == 0 and epoch > test_interval:
                train_loss  = self.nn_loss(self.params, self.inputs_train, self.targets_train)
                test_loss = self.nn_loss(self.params, self.inputs_test, self.targets_test)
                
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                print(f"Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")
                # Save the model parameters
                model_path= os.path.join(current_dir, 'models')
                os.mkdir(model_path) if not os.path.exists(model_path) else None 

                # Plotting the error vs epochs
                plt.figure()
                plt.plot(range(test_interval, epoch, test_interval), train_losses, label='Train Loss')
                plt.plot(range(test_interval, epoch, test_interval), test_losses, label='Test Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.title(f'system_id_nn_{run_name} Error vs Epochs')
 
                # Save the plot
                plot_path = os.path.join(current_dir, 'plots')
                if not os.path.exists(plot_path):
                    os.makedirs(plot_path)
                plot_filename = os.path.join(plot_path, f'system_id_nn_{run_name}_{epochs}eps_error_vs_epochs_{timestamp}.png')
                plt.savefig(plot_filename)
                plt.close()

        with open(os.path.join(model_path, f'system_id_nn_{run_name}_{epochs}eps_{timestamp}.pt'), 'wb') as f:
            pickle.dump(self.params, f)
        
    def gp_loss(self, params, inputs, targets):
        kernel = params[0] * jnp.exp(-0.5 * params[1] * jnp.sum((inputs[:, None, :] - inputs[None, :, :]) ** 2, axis=-1))
        K = kernel + params[2] * jnp.eye(len(inputs))
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, targets))
        preds = jnp.dot(kernel, alpha)
        return jnp.mean((preds - targets) ** 2)
    

    def optimize_gp(self, run_name: str = ''):
        initial_params_list = [
            jnp.array([0.5, 0.5, 1e-6]),
            jnp.array([2.0, 2.0, 1e-6]),
            jnp.array([5.0, 5.0, 1e-6]),
            jnp.array([20.0, 20.0, 1e-6]),
            jnp.array([50.0, 50.0, 1e-6]),
            jnp.array([100.0, 100.0, 1e-6]),
            jnp.array([200.0, 200.0, 1e-6]),
            jnp.array([500.0, 500.0, 1e-6]),
            jnp.array([1000.0, 1000.0, 1e-6]),
            jnp.array([0.1, 10.0, 1e-6]),
            jnp.array([10.0, 0.1, 1e-6]),
            jnp.array([0.1, 0.1, 1e-6]),
            jnp.array([1.0, 0.5, 1e-6]),
            jnp.array([0.5, 1.0, 1e-6]),
            jnp.array([2.0, 0.5, 1e-6]),
            jnp.array([0.5, 2.0, 1e-6]),
            jnp.array([5.0, 0.5, 1e-6]),
            jnp.array([0.5, 5.0, 1e-6]),
            jnp.array([20.0, 0.5, 1e-6]),
            jnp.array([0.5, 20.0, 1e-6])
        ]
        final_test_losses = []

        for initial_params in initial_params_list:
            test_loss = self.iterative_gp_optimization(initial_params=initial_params, max_iterations=20, data_increment=10)
            final_test_losses.append((initial_params, test_loss))
            print(f"Initial Params: {initial_params}, Final Test Loss: {test_loss}")

        # Visualize the relationship between initial_params and final test_loss
        plt.figure()
        for initial_params, test_loss in final_test_losses:
            plt.scatter(str(initial_params), test_loss, label=f'Initial Params: {initial_params}')
        plt.xlabel('Initial Parameters')
        plt.ylabel('Final Test Loss')
        plt.title('Initial Parameters vs Final Test Loss')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(current_dir, 'plots')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_filename = os.path.join(plot_path, f'system_id_gp_{run_name}_initial_params_vs_test_loss_{timestamp}.png')
        plt.savefig(plot_filename)
        plt.close()

    def iterative_gp_optimization(self, run_name: str = '', initial_params: jnp.ndarray = jnp.array([1.0  * 1e6, 1.0 * 1e6, 1e-6 * 1e10]), max_iterations=20, data_increment=10):
        initial_data_size = min(len(self.inputs_train), data_increment)
        data_sizes = []
        test_losses = []
        max_iterations = min(max_iterations, int(len(self.inputs_train) / initial_data_size))
        for i in range(1, max_iterations + 1):
            current_data_size = min(data_increment * i, len(self.inputs_train))
            data_sizes.append(current_data_size)

            # Subset the training data
            inputs_subset = self.inputs_train[:current_data_size]
            targets_subset = self.targets_train[:current_data_size]

            # Optimize GP with the current subset
            result = minimize(self.gp_loss, initial_params, args=(inputs_subset, targets_subset), method='L-BFGS-B')

            # Test the GP with the test set
            optimized_params = result.x
            test_loss = self.gp_loss(optimized_params, self.inputs_test, self.targets_test)
            test_losses.append(test_loss)
            
            print(f"Iteration {i}, Data Size: {current_data_size}, Test Loss: {test_loss}")

            # Save the model parameters
            #model_path= os.path.join(current_dir, 'models')
            #os.mkdir(model_path) if not os.path.exists(model_path) else None
            #with open(os.path.join(model_path, f'system_id_gp_{run_name}_iteration_{i}_{timestamp}.pt'), 'wb') as f:
            #    pickle.dump(optimized_params, f)
            

        # Plotting the data volume vs error
        plt.figure()
        plt.plot(data_sizes, test_losses, label='Test Loss')
        plt.xlabel('Data Volume')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'system_id_gp_{run_name} initial params: {initial_params}')

        # Save the plot 
        plot_path = os.path.join(current_dir, 'plots')
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plot_filename = os.path.join(plot_path, f'system_id_gp_{run_name} initial params: {initial_params}.png')
        plt.savefig(plot_filename)
        plt.close()

        return test_losses[-1]

    def run(self, run_name: str = 'system_id'):
        self.load_data()
        
        self.train_nn(run_name)
        # self.test_nn()
        # gp_result = self.optimize_gp()
        #self.optimize_gp(run_name)
        print("NN and GP System ID completed.")
        

if __name__ == "__main__":
    system_id = SystemID()
    system_id.run()
