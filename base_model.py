import tensorflow as tf
from tensorflow.keras.layers import Conv1D,Conv1DTranspose,Concatenate,Input

def create_model(batching_size=12000, actv="relu"):
    inp = Input(shape=(batching_size,1))
    conv1 = Conv1D(2,32,2,'same',activation=actv)(inp)
    conv2 = Conv1D(4,32,2,'same',activation=actv)(conv1)
    conv3 = Conv1D(8,32,2,'same',activation=actv)(conv2)
    conv4 = Conv1D(16,32,2,'same',activation=actv)(conv3)
    conv5 = Conv1D(32,32,2,'same',activation=actv)(conv4)

    dconv1 = Conv1DTranspose(32,32,1,padding='same')(conv5)
    conc = Concatenate()([conv5,dconv1])
    dconv2 = Conv1DTranspose(16,32,2,padding='same')(conc)
    conc = Concatenate()([conv4,dconv2])
    dconv3 = Conv1DTranspose(8,32,2,padding='same')(conc)
    conc = Concatenate()([conv3,dconv3])
    dconv4 = Conv1DTranspose(4,32,2,padding='same')(conc)
    conc = Concatenate()([conv2,dconv4])
    dconv5 = Conv1DTranspose(2,32,2,padding='same')(conc)
    conc = Concatenate()([conv1,dconv5])
    dconv6 = Conv1DTranspose(1,32,2,padding='same')(conc)
    conc = Concatenate()([inp,dconv6])
    dconv7 = Conv1DTranspose(1,32,1,padding='same',activation='linear')(conc)
    model = tf.keras.models.Model(inp,dconv7)
    return model