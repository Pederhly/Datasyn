import utils
import numpy as np
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    """
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    """
    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!
    shuffle_data = True
    use_improved_sigmoid = True   
    use_improved_weight_init = True
    use_momentum = True

    model_no = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no, val_history_no = trainer_shuffle.train(
        num_epochs)

    neurons_per_layer = np.append(64*np.ones(10), 10).astype(int) # for task 4e
    shuffle_data = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate = .02

    model_yes = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_yes, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_yes, val_history_yes = trainer_shuffle.train(
        num_epochs)

    
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history_no["loss"], "Task 3, Training", npoints_to_average=10)
    utils.plot_loss(train_history_yes["loss"], "Task 4e, Training", npoints_to_average=10)
    utils.plot_loss(val_history_no["loss"], "Task 3, Validation", npoints_to_average=10)
    utils.plot_loss(val_history_yes["loss"], "Task 4e, Validation", npoints_to_average=10)
    plt.ylabel("Loss")
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .99])
    utils.plot_loss(train_history_no["accuracy"], "Task 3, Training")
    utils.plot_loss(train_history_yes["accuracy"], "Task 4e, Training")
    utils.plot_loss(val_history_no["accuracy"], "Task 3, Validation")
    utils.plot_loss(val_history_yes["accuracy"], "4e, Validation")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()
    plt.savefig("task3.png")
    """
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history_no["loss"], "Task 2 Model, Training", npoints_to_average=10)
    utils.plot_loss(train_history_yes["loss"], "With improved sigmoid, weight and momentum, Training", npoints_to_average=10)
    utils.plot_loss(val_history_no["loss"], "Task 2 Model, Validation", npoints_to_average=10)
    utils.plot_loss(val_history_yes["loss"], "With improved sigmoid, weight and momentum Validation", npoints_to_average=10)
    plt.ylabel("Loss")
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .99])
    utils.plot_loss(train_history_no["accuracy"], "Task 2 Model, Training")
    utils.plot_loss(train_history_yes["accuracy"], "With improved sigmoid, weight and momentum, Training")
    utils.plot_loss(val_history_no["accuracy"], "Task 2 Model, Validation")
    utils.plot_loss(val_history_yes["accuracy"], "With improved sigmoid, weight and momentum, Validation")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()
    plt.savefig("task3.png")
    """