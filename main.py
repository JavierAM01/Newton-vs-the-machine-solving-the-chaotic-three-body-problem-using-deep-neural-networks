import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data():
    X = pd.read_csv("data/train_X.csv")
    y = pd.read_csv("data/train_Y.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42) # 1% testing
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

def create_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=3))  # Input layer with the first hidden layer
    for _ in range(9):  # Remaining 9 hidden layers
        model.add(Dense(128, activation='relu'))
    model.add(Dense(4))  # Output layer
    return model

def train(model, X_train, y_train, validation_data=None):
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='mae')
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=validation_data,
        epochs=1000,
        batch_size=5000,
        verbose=1
    )
    
    return history

def plot_trajectory(model, x1, x2, name="example", save=True):
    global X_train, X_test, y_train, y_test
    #
    # get the trajectory points
    index_train = (X_train.iloc[:,1] == x1) & (X_train.iloc[:,2] == x2)
    index_test  = (X_test.iloc[:,1] == x1)  & (X_test.iloc[:,2] == x2)
    X1 = pd.concat([X_train[index_train], X_test[index_test]])
    y1 = pd.concat([y_train[index_train], y_test[index_test]])
    #
    # sort the trajectory by time
    index = np.argsort(X1.iloc[:,0].values)
    X1 = X1.iloc[index,:]
    y1 = y1.iloc[index,:]
    #
    # make predictions
    pred = model.predict(X1) 
    pred = pd.DataFrame(pred)
    #
    # plot trajectory
    def plot_T(data, alpha=1):
        plt.scatter([data.iloc[0,0], data.iloc[0,2], -data.iloc[0,0]-data.iloc[0,2]], [data.iloc[0,1], data.iloc[0,3], -data.iloc[0,1]-data.iloc[0,3]], c=["r", "b", "g"])
        plt.plot(data.iloc[:,0], data.iloc[:,1], "r-", alpha=alpha)
        plt.plot(data.iloc[:,2], data.iloc[:,3], "b-", alpha=alpha)
        plt.plot(-data.iloc[:,0]-data.iloc[:,2], -data.iloc[:,1]-data.iloc[:,3], "g-", alpha=alpha)
        plt.xticks([])
        plt.yticks([])
    #
    plot_T(y1, alpha=0.5)
    plot_T(pred, alpha=1)
    #
    if save:
        plt.savefig(f"{name}.png")
        plt.clf()


def animation(model, x1, x2, name="0"):
    """
        The same idea from plot trajectory, but instead we are going to create an animation in 
        real time. This is going to be made by a tail (alpha=0.5) of the real brutus numerical 
        approximation and a circle representing the ANN current prediction. Our goal is to have the circle 
        as closer as possible to the end of the tail, which would mean a "perfect" prediction. 
    """
    global X_train, X_test, y_train, y_test
    #
    # get the trajectory points
    index_train = (X_train.iloc[:,1] == x1) & (X_train.iloc[:,2] == x2)
    index_test  = (X_test.iloc[:,1] == x1)  & (X_test.iloc[:,2] == x2)
    X1 = pd.concat([X_train[index_train], X_test[index_test]])
    y1 = pd.concat([y_train[index_train], y_test[index_test]])
    #
    # sort the trajectory by time
    index = np.argsort(X1.iloc[:,0].values)
    X1 = X1.iloc[index,:]
    y1 = y1.iloc[index,:]
    #
    # make predictions
    pred = model.predict(X1) 
    pred = pd.DataFrame(pred)
    #
    import matplotlib.animation as animation
    #
    # Define the plot function
    def plot_T(t, pred, y1):
        ax.clear()
        plt.scatter([pred.iloc[t, 0], pred.iloc[t, 2], -pred.iloc[t, 0] - pred.iloc[t, 2]], 
                    [pred.iloc[t, 1], pred.iloc[t, 3], -pred.iloc[t, 1] - pred.iloc[t, 3]], 
                    c=["r", "b", "g"])
        plt.plot(y1.iloc[:t, 0], y1.iloc[:t, 1], "r-", alpha=0.5)
        plt.plot(y1.iloc[:t, 2], y1.iloc[:t, 3], "b-", alpha=0.5)
        plt.plot(-y1.iloc[:t, 0] - y1.iloc[:t, 2], -y1.iloc[:t, 1] - y1.iloc[:t, 3], "g-", alpha=0.5)
        plt.xticks([])
        plt.yticks([])
    #
    # Create the animation
    fig, ax = plt.subplots()
    def update(frame):
        plot_T(frame, pred, y1)
    #
    ani = animation.FuncAnimation(fig, update, frames=len(y1), repeat=False, interval=30)
    ani.save(f"animation_{name}.gif", writer="pillow")


def get_sample(X):
    n = len(X)
    i = np.random.randint(0,n)
    x1, x2 = X.iloc[i,1], X.iloc[i,2]    
    return x1, x2


def plot_error(history):
    plt.clf()
    values = list(history.history['loss'])[::10]
    plt.plot(values, "r-")
    values2 = list(history.history['val_loss'])[::10]
    plt.plot(values2, "r--", alpha=0.5)
    plt.title("")
    plt.yscale("log")
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    x = np.array(range(0,101,20))
    plt.xticks(x, labels=10*x)
    plt.savefig(f"mae.png")

def plot_results(model, seed=42):
    """
        Here we'll plot 2 trajectories from the training set at the left and 2 more from 
        the validation set at the right. 

        Comments:
        ---------
        
        Note that "from the validation set" in this case is
        not completely true as the way our dataset is structure and how we split it, probably 
        we are not going to have a complete trajectory just in the validation data, instead we 
        are going to have some of the points from one trajectory in the training set and others in
        the validation set. 
        
        Knowing this, the selection I'm going to do it the followinfg way: chose one point from the 
        data set that I want (that way I already know that some of those points are going to be there)
        and then to plot the final trajectory I'll join the points from that trajectory from both datsets.

        Alternatives:
        ------------- 
        
        One way to solve this globaly is to do the split the correct way. First we get all the 
        trajectories and take randomly 1% of them for validation. Then filter all those points who are in 
        the training / validation part to separate the final dataset. That's not very efficient as we have
        to go one by one in a big dataset, and not just a simple comparison one to one, each of them has to do a 
        "in" comparison in which we check if that trajectory is in the corresponding set. For this case I would 
        use spark for a faster execution (until now pandas was enough as we just had to read the data).
    """
    global X_train, X_test, y_train, y_test
    np.random.seed(seed)
    plt.figure(figsize=(10,6))
    plt.suptitle("ANN trajectories predictions\n - Train vs Test - ")
    for i in range(2):
        plt.subplot(2, 2, 2*i+1)
        x1, x2 = get_sample(X_train)
        plot_trajectory(model, x1, x2, save=False)
        animation(model, x1, x2, name=str(2*i+1))
    for i in range(2):
        plt.subplot(2, 2, 2*i+2)
        x1, x2 = get_sample(X_test)
        plot_trajectory(model, x1, x2, save=False)
        animation(model, x1, x2, name=str(2*i+2))
    plt.savefig("trajectories.png")


if __name__ == "__main__":
    model = create_model()
    history = train(model, X_train, y_train, validation_data=(X_test, y_test))
    plot_error(history)
    plot_results(model)

