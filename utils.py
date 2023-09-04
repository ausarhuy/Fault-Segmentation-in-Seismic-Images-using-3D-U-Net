import numpy as np
import matplotlib.pyplot as plt


def data_norm(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std


def plot2d(seis, fault, pred=None, at=1):
    # Create figure and axes
    fig = plt.figure(figsize=(15, 5))
    #
    seis_slice_ax = fig.add_subplot(131)
    seis_slice_ax.set_title('Seismic 2D sliced image')
    seis_slice_ax.imshow(seis, vmin=-2, vmax=2, cmap=plt.cm.bone, interpolation='nearest', aspect=at)
    #
    fault_slice_ax = fig.add_subplot(132)
    fault_slice_ax.imshow(fault, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='nearest', aspect=at)
    fault_slice_ax.set_title('Fault segment Growth-Truth 2D sliced image')
    #
    if pred is not None:
        fault_slice_ax = fig.add_subplot(133)
        fault_slice_ax.imshow(fault, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='nearest', aspect=at)
        fault_slice_ax.set_title('Fault segment Prediction 2D sliced image')

    # Adjust spacing between subplots
    fig.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.show()


def plot3d(seis, fault, pred=None):
    # Create figure and 3D axes
    fig = plt.figure(figsize=(15, 5))
    seis_ax = fig.add_subplot(121, projection='3d')
    fault_ax = fig.add_subplot(122, projection='3d')
    # Generate coordinates for each point in the volume
    x, y, z = np.meshgrid(np.arange(seis.shape[0]), np.arange(seis.shape[1]), np.arange(seis.shape[2]))
    # coordinates for plotting
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    # Create a scatter plot for seismic image
    seis_ax.scatter(x, y, z, c=seis.flatten(), cmap=plt.cm.bone, marker='.')
    seis_ax.set_xlabel('X')
    seis_ax.set_ylabel('Y')
    seis_ax.set_zlabel('Z')
    seis_ax.set_title('Seismic 3D image')
    # Create a scatter plot for fault segment
    fault_ax.scatter(x, y, z, c=fault.flatten(), cmap=plt.cm.bone, marker='.')
    fault_ax.set_xlabel('X')
    fault_ax.set_ylabel('Y')
    fault_ax.set_zlabel('Z')
    fault_ax.set_title('Fault segment 3D image')
    # Create a scatter plot for fault segment prediction
    if pred is not None:
        pred_ax = fig.add_subplot(123, projection='3d')
        pred_ax.scatter(x, y, z, c=pred.flatten(), cmap=plt.cm.bone, marker='.')
        pred_ax.set_xlabel('X')
        pred_ax.set_ylabel('Y')
        pred_ax.set_zlabel('Z')
        pred_ax.set_title('Fault segment Prediction 3D image')

    # Adjust spacing between subplots
    fig.subplots_adjust(wspace=0.3)
    plt.tight_layout()
    plt.show()


def show_history(history, save_path=None):
    plt.figure(figsize=(10, 6))
    # summarize history for accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('Model accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'validation'], loc='center right', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    if save_path is not None:
        plt.savefig(save_path+'accuracy.png')
    plt.show()

    # summarize history for loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(['train', 'validation'], loc='center right', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    if save_path is not None:
        plt.savefig(save_path+'lose.png')
    plt.show()


def show_histories(histories, save_path=None):
    plt.figure(figsize=(10, 6))

    # Define line colors
    colors = ['blue', 'red', 'green', 'orange']

    # Plot accuracy for each history with different line colors
    for i, (name, history) in enumerate(histories.items()):
        plt.plot(history.history['binary_accuracy'], label=f'{name} train', color=colors[i % len(colors)])

    # Plot validation accuracy for each history with different line colors
    for i, (name, history) in enumerate(histories.items()):
        plt.plot(history.history['val_binary_accuracy'], label=f'{name} validation', linestyle='dashed',
                 color=colors[i % len(colors)])

    plt.title('Models accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(loc='center right', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    if save_path is not None:
        plt.savefig(save_path+'accuracies.png')
    plt.show()

    plt.figure(figsize=(10, 6))

    # Plot loss for each history with different line colors
    for i, (name, history) in enumerate(histories.items()):
        plt.plot(history.history['loss'], label=f'{name} train', color=colors[i % len(colors)])

    # Plot validation loss for each history with different line colors
    for i, (name, history) in enumerate(histories.items()):
        plt.plot(history.history['val_loss'], label=f'{name} validation', linestyle='dashed',
                 color=colors[i % len(colors)])

    plt.title('Models loss', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.legend(loc='center right', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    if save_path is not None:
        plt.savefig(save_path+'loses.png')
    plt.show()
