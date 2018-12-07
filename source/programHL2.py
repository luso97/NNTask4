import tensorflow as tf
import numpy as np
import numpy.random as rd
from nn18_ex2_load import load_isolet
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import transpose



Xa,C,X_test,C_test=load_isolet();
print(Xa.shape);
print(C.shape);
print(X_test.shape);
print(C_test.shape);


#We create the C arrays with this function
def createC(array):
    res=np.zeros((len(array),26));
    
    for i in range(len(array)):
        res[i][array[i]-1]=1;
    return res;
#normalize function

def normalizeDataNN(array):
    m=transpose(array);
    res=np.zeros((len(array[0]),len(array)),dtype=float);
    for i in range (len(m)):
        maximum=max(m[i]);
        minimum=min(m[i]);
        for j in range(len(m[i])):
            res[i][j]=float(m[i][j]-minimum)/float(maximum-minimum);
    return transpose(res);

Ca=createC(C);
Ca_Test=createC(C_test);
Xa=normalizeDataNN(Xa);
Ca=normalizeDataNN(Ca);
X_test=normalizeDataNN(X_test);
Ca_Test=normalizeDataNN(Ca_Test);

# Give the dimension of the data and chose the number of hidden layer

n_in = 300
n_hidden = 40
n_hidden = 75
n_hidden3 = 5
n_out = 26
x = tf.placeholder(shape=(None,300),dtype=tf.float64)
# Set the variables
W_hid = tf.Variable(rd.randn(n_in,n_hidden) / np.sqrt(n_in),trainable=True)
b_hid = tf.Variable(np.zeros(n_hidden),trainable=True);
y1=tf.nn.relu(tf.matmul(x,W_hid) + b_hid);
# Lets try adding another hidden layer (if we want it, decomment the line)
W_hid2 = tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True);
b_hid2 = tf.Variable(np.zeros(n_hidden),trainable=True)
y2=tf.nn.relu(tf.matmul(y1,W_hid2)+b_hid2);
# And another one (if we want it, decomment the line)
W_hid3 = tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True);
b_hid3 = tf.Variable(np.zeros(n_hidden),trainable=True)
y3=tf.nn.relu(tf.matmul(y2,W_hid3)+b_hid3);

W_hid4 = tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True);
b_hid4 = tf.Variable(np.zeros(n_hidden),trainable=True)
y4=tf.nn.relu(tf.matmul(y3,W_hid4)+b_hid4);

W_hid5 = tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True);
b_hid5 = tf.Variable(np.zeros(n_hidden),trainable=True)
y5=tf.nn.relu(tf.matmul(y4,W_hid5)+b_hid5);

W_hid6 = tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True);
b_hid6 = tf.Variable(np.zeros(n_hidden),trainable=True)
y6=tf.nn.relu(tf.matmul(y5,W_hid6)+b_hid6);

W_hid7 = tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True);
b_hid7 = tf.Variable(np.zeros(n_hidden),trainable=True)
y7=tf.nn.relu(tf.matmul(y6,W_hid7)+b_hid7);

W_hid8 = tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True);
b_hid8 = tf.Variable(np.zeros(n_hidden),trainable=True)
y8=tf.nn.relu(tf.matmul(y7,W_hid8)+b_hid8);

W_hid9 = tf.Variable(rd.randn(n_hidden,n_hidden) / np.sqrt(n_in),trainable=True);
b_hid9 = tf.Variable(np.zeros(n_hidden),trainable=True)
y9=tf.nn.relu(tf.matmul(y8,W_hid9)+b_hid9);
####################################### 
### For 1 hidden layer, decomment the 1st w_out, for 2 the 2nd and for 3 the 3rd (keep the other commented)
w_out = tf.Variable(rd.randn(n_hidden,n_out) / np.sqrt(n_in),trainable=True)
#w_out = tf.Variable(rd.randn(n_hidden,n_out) / np.sqrt(n_in),trainable=True)
Tw_out = tf.Variable(rd.randn(n_hidden3,n_out) / np.sqrt(n_in),trainable=True)
#######################################
b_out = tf.Variable(np.zeros(n_out),trainable=True)

# Define the neuron operations

####################################### 
### For 1 hidden layer, decomment the 1st y, for 2 the 2nd and for 3 the 3rd (keep the other commented)

y = y9
#y = tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(tf.nn.relu(tf.matmul(x,W_hid) + b_hid),W_hid2)+b_hid2),W_hid3)+b_hid3)
z = tf.nn.softmax(tf.matmul(y,w_out) + b_out)



z_ = tf.placeholder(shape=(None,26),dtype=tf.float64)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(tf.clip_by_value(z,1e-10,1.0)), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(z_ * tf.log(z), reduction_indices=[1]));


train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(z,1), tf.argmax(z_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
init = tf.global_variables_initializer() # Create an op that will
sess = tf.Session()
sess.run(init) # Set the value of the variables to their initialization value



# Re init variables to start from scratch
sess.run(init)

# Create some list to monitor how error decreases
test_loss_list = []
train_loss_list = []

test_acc_list = []
train_acc_list = []

# Create minibtaches to train faster
k_batch = 40
X_batch_list = np.array_split(Xa,k_batch)
labels_batch_list = np.array_split(Ca,k_batch)
initial=np.random.randint(0,6238-100);
validation_set=Xa[initial:initial+100][:];
Ca_validation_set=Ca[initial:initial+100][:]
maxacc=0.0;
trainacc=0.0;
saver = tf.train.Saver()
epochs=0;
counter=0;
patience=100;
#Training with validation set
for k in range(500):
    
    # Run gradient steps over each minibatch
    for x_minibatch,labels_minibatch in zip(X_batch_list,labels_batch_list):
        sess.run(train_step, feed_dict={x: x_minibatch, z_:labels_minibatch})
    counter=counter+1;
    
    # Compute the errors over the whole dataset
    train_loss = sess.run(cross_entropy, feed_dict={x:Xa, z_:Ca})
    test_loss = sess.run(cross_entropy, feed_dict={x:validation_set,z_:Ca_validation_set});
    
    # Compute the acc over the whole dataset
    train_acc = sess.run(accuracy, feed_dict={x:Xa, z_:Ca})
    test_acc = sess.run(accuracy, feed_dict={x:validation_set,z_:Ca_validation_set});
    if maxacc<test_acc:
        save_path = saver.save(sess, "/tmp/model.ckpt")
        maxacc=test_acc;
        trainacc=train_acc;
        epochs=k;
        
        #Save state
    
    if counter>patience:
        print("early stopping");
        break;
    # Put it into the lists
    test_loss_list.append(test_loss)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)
    if k>0 and (test_acc_list[k]-test_acc_list[k-1])<0.01:
        counter+=1;
    else:
        counter=0;
    
    if np.mod(k,100) == 0:
        print('iteration {} test accuracy: {:.3f}'.format(k+1,test_acc))
print('best val accuracy: '+format(maxacc));
print('best train accuracy: '+format(trainacc));
fig,ax_list = plt.subplots(1,2)
ax_list[0].plot(train_loss_list, color='blue', label='training', lw=2)
ax_list[0].plot(test_loss_list, color='green', label='testing', lw=2)
ax_list[1].plot(train_acc_list, color='blue', label='training', lw=2)
ax_list[1].plot(test_acc_list, color='green', label='testing', lw=2)

ax_list[0].set_xlabel('training iterations')
ax_list[1].set_xlabel('training iterations')
ax_list[0].set_ylabel('Cross-entropy')
ax_list[1].set_ylabel('Accuracy')
plt.legend(loc=2)
plt.show();
#Now we use the test set
test_loss_list2 = []
train_loss_list2 = []

test_acc_list2 = []
train_acc_list2 = []

saver.restore(sess, "/tmp/model.ckpt")
saver2 = tf.train.Saver()
print("epochs"+str(epochs));
counter=0;
patience=1000;
maxacc=0;
#retraining of the network with the epoch that suits best
for k in range(epochs):
    
    # Run gradient steps over each minibatch
    for x_minibatch,labels_minibatch in zip(X_batch_list,labels_batch_list):
        sess.run(train_step, feed_dict={x: x_minibatch, z_:labels_minibatch})
    
    
    # Compute the errors over the whole dataset
    train_loss = sess.run(cross_entropy, feed_dict={x:Xa, z_:Ca})
    test_loss = sess.run(cross_entropy, feed_dict={x:validation_set,z_:Ca_validation_set});
    
    # Compute the acc over the whole dataset
    train_acc = sess.run(accuracy, feed_dict={x:Xa, z_:Ca})
    test_acc = sess.run(accuracy, feed_dict={x:X_test,z_:Ca_Test});
    if maxacc<test_acc:
        maxacc=test_acc;
        save_path = saver.save(sess, "/tmp/model2.ckpt")
        #Save state 
    # Put it into the lists
    test_loss_list2.append(test_loss)
    train_loss_list2.append(train_loss)
    test_acc_list2.append(test_acc)
    train_acc_list2.append(train_acc)
    if k>0 and (test_acc_list2[k]-test_acc_list2[k-1])<0.02:
        counter+=1;
    else:
        counter=0;
    if counter>patience:
        print("early stopping");
        break;
    
    if np.mod(k,100) == 0:
        print('iteration {} test accuracy: {:.3f}'.format(k+1,test_acc))
fig,ax_list = plt.subplots(1,2)
ax_list[0].plot(train_loss_list2, color='blue', label='training', lw=2)
ax_list[0].plot(test_loss_list2, color='green', label='testing', lw=2)
ax_list[1].plot(train_acc_list2, color='blue', label='training', lw=2)
ax_list[1].plot(test_acc_list2, color='green', label='testing', lw=2)

ax_list[0].set_xlabel('training iterations')
ax_list[1].set_xlabel('training iterations')
ax_list[0].set_ylabel('Cross-entropy')
ax_list[1].set_ylabel('Accuracy')
plt.legend(loc=2)

plt.show();
print('best test set accuracy: {:.3f}'.format(maxacc));
