import random
import numpy as np
import matplotlib.pyplot as plt


def get_random_states():
    states = []

    # PyTorch random generator state
    try:
        import torch
        torch_state = torch.get_rng_state().numpy().tobytes().hex()
        states.append(f"PyTorch: {torch_state}")
    except ImportError:
        pass

    # Python random generator state
    python_random_state = random.getstate()
    python_random_state_str = '|'.join(map(str, python_random_state[1])) + '|' + str(python_random_state[0])
    states.append(f"Python Random: {python_random_state_str}")

    # NumPy random generator state
    try:
        numpy_random_state = np.random.get_state()
        numpy_random_state_str = '|'.join(map(str, numpy_random_state[1])) + '|' + numpy_random_state[0].tobytes().hex()
        states.append(f"NumPy: {numpy_random_state_str}")
    except AttributeError:
        pass

    # Matplotlib random generator state
    try:
        mpl_state = plt.matplotlib.get_random_seed()
        states.append(f"Matplotlib: {mpl_state}")
    except AttributeError:
        pass

    # TensorFlow random generator state
    try:
        import tensorflow as tf
        tf_state = tf.random.get_global_generator().state.numpy().tobytes().hex()
        states.append(f"TensorFlow: {tf_state}")
    except ImportError:
        pass

    return '; '.join(states)


def restore_random_states(random_states_str):
    states = {}
    for line in random_states_str.split(';'):
        lib_name, lib_state = line.split(': ')
        try:
            if lib_name == 'PyTorch':
                try:
                    import torch
                    # Restore PyTorch random generator state
                    torch_state = bytes.fromhex(lib_state)
                    torch.set_rng_state(torch_state)
                except ImportError:
                    pass
            elif lib_name == 'Python Random':
                # Restore Python random generator state
                python_random_state_str = lib_state.split('|')
                python_random_state = (int(python_random_state_str[-1]), tuple(map(int, python_random_state_str[:-1])))
                random.setstate(python_random_state)
            elif lib_name == 'NumPy':
                try:
                    import numpy as np
                    # Restore NumPy random generator state
                    numpy_random_state_str = lib_state.split('|')
                    numpy_random_state = (tuple(map(int, numpy_random_state_str[:-1])),
                                          np.frombuffer(bytes.fromhex(numpy_random_state_str[-1]), dtype=np.uint32))
                    np.random.set_state(numpy_random_state)
                except ImportError:
                    pass
            elif lib_name == 'Matplotlib':
                try:
                    import matplotlib.pyplot as plt
                    # Restore Matplotlib random generator state
                    plt.matplotlib.random.seed(int(lib_state))
                except ImportError:
                    pass
            elif lib_name == 'TensorFlow':
                try:
                    import tensorflow as tf
                    # Restore TensorFlow random generator state
                    tf_state = bytes.fromhex(lib_state)
                    tf.random.set_global_generator(tf.random.Generator.from_state(tf_state))
                except ImportError:
                    pass
        except ImportError:
            pass

    # Return a dictionary containing restored random states
    return states


if __name__ == "__main__":
    random_states_str = get_random_states()
    print(random_states_str)
