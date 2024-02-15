import numpy as np
import stim
import pymatching

circuit = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=0.005)

model = circuit.detector_error_model(decompose_errors=True)
matching = pymatching.Matching.from_detector_error_model(model)

sampler = circuit.compile_detector_sampler()

syndrome, actual_observables = sampler.sample(shots=1000, separate_observables=True)

num_errors = 0
for i in range(syndrome.shape[0]):
    predicted_observables = matching.decode(syndrome[i, :])
    num_errors += not np.array_equal(actual_observables[i, :], predicted_observables)

print(num_errors)
print()
print(syndrome, actual_observables)
