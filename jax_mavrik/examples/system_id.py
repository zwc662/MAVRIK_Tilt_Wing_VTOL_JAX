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



class SystemID:
    def __init__(self, data_path):
        self.data_path = data_path
        self.states = None
        self.controls = None
        self.params = None
        self.predict = None
        self.scaler = StandardScaler()
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(0.001)
        self.opt_state = None

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
        pt_files = glob.glob(os.path.join(self.data_path, '*.pt'))
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

    def loss(self, params, inputs, targets):
        preds = self.predict(params, inputs)
        return jnp.mean((preds - targets) ** 2)

    def update(self, params, inputs, targets, lr):
        grads = jax.grad(self.loss)(params, inputs, targets)
        self.opt_state = self.opt_update(0, grads, self.opt_state)
        return self.get_params(self.opt_state)
  
    def train_nn(self, epochs=1000_000_000, lr=0.001, test_interval=100):
        train_losses = []
        test_losses = []
        for epoch in range(epochs):
            self.params = self.update(self.params, self.inputs_train, self.targets_train, lr)
            if epoch % test_interval == 0:
                train_loss = self.loss(self.params, self.inputs_train, self.targets_train)
                test_loss = self.loss(self.params, self.inputs_test, self.targets_test)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                print(f"Epoch {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")

        # Plotting the error vs epochs
        plt.figure()
        plt.plot(range(0, epochs, test_interval), train_losses, label='Train Loss')
        plt.plot(range(0, epochs, test_interval), test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Error vs Epochs')

        # Save the plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join('plots', f'error_vs_epochs_{timestamp}.png')
        plt.savefig(plot_filename)
        plt.close()

    def gp_loss(self, params, inputs, targets):
        kernel = params[0] * jnp.exp(-0.5 * params[1] * jnp.sum((inputs[:, None, :] - inputs[None, :, :]) ** 2, axis=-1))
        K = kernel + params[2] * jnp.eye(len(inputs))
        L = jnp.linalg.cholesky(K)
        alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, targets))
        preds = jnp.dot(kernel, alpha)
        return jnp.mean((preds - targets) ** 2)
 
    def iterative_gp_optimization(self, max_iterations=10, data_increment=0.1):
        initial_data_size = int(len(self.inputs_train) * data_increment)
        data_sizes = []
        test_losses = []

        for i in range(1, max_iterations + 1):
            current_data_size = min(initial_data_size * i, len(self.inputs_train))
            data_sizes.append(current_data_size)

            # Subset the training data
            inputs_subset = self.inputs_train[:current_data_size]
            targets_subset = self.targets_train[:current_data_size]

            # Optimize GP with the current subset
            initial_params = jnp.array([1.0, 1.0, 1e-6])
            result = minimize(self.gp_loss, initial_params, args=(inputs_subset, targets_subset), method='L-BFGS-B')

            # Test the GP with the test set
            optimized_params = result.x
            test_loss = self.gp_loss(optimized_params, self.inputs_test, self.targets_test)
            test_losses.append(test_loss)

            print(f"Iteration {i}, Data Size: {current_data_size}, Test Loss: {test_loss}")

        # Plotting the data volume vs error
        plt.figure()
        plt.plot(data_sizes, test_losses, label='Test Loss')
        plt.xlabel('Data Volume')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Data Volume vs Error')

        # Save the plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join('plots', f'data_volume_vs_error_{timestamp}.png')
        plt.savefig(plot_filename)
        plt.close()

    def run(self):
        self.load_data()
        self.initialize_nn()
        self.train_nn()
        # self.test_nn()
        # gp_result = self.optimize_gp()
        self.iterative_gp_optimization()
        print("NN and GP System ID completed.")
        

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs('plots', exist_ok=True)
    system_id = SystemID(data_path)
    system_id.run()
