from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import InputLayer, ConcatLayer, Conv2DLayer

input_var = T.tensor4('inputs')
input_var_ex = T.ivector('input_var_ex')

def ConvFactory(data, num_filter, filter_size, stride=1, pad=0, nonlinearity=lasagne.nonlinearities.leaky_rectify):
    data = lasagne.layers.batch_norm(Conv2DLayer(
        data, num_filters=num_filter,
        filter_size=filter_size,
        stride=stride, pad=pad,
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(gain='relu')))
    return data

def SimpleFactory(data, ch_1x1, ch_3x3):
    conv1x1 = ConvFactory(data=data, filter_size=1, pad=0, num_filter=ch_1x1)
    conv3x3 = ConvFactory(data=data, filter_size=3, pad=1, num_filter=ch_3x3) 
    concat = ConcatLayer([conv1x1, conv3x3])
    return concat

input_shape = (None, channels, framesize, framesize)
img = InputLayer(shape=input_shape, input_var=input_var[input_var_ex])
net = img


net = ConvFactory(net, filter_size=3, num_filter=64, pad=patch_size)
print(net.output_shape)
net = SimpleFactory(net, 16, 16)
print(net.output_shape)
net = SimpleFactory(net, 16, 32)
print(net.output_shape)
net = ConvFactory(net, filter_size=14, num_filter=16) 
print(net.output_shape)
net = SimpleFactory(net, 136, 32)
print(net.output_shape)
net = SimpleFactory(net, 64, 64)
print(net.output_shape)
net = SimpleFactory(net, 64, 32)
print(net.output_shape)
net = SimpleFactory(net, 40, 96)
print(net.output_shape)
net = ConvFactory(net, filter_size=18, num_filter=32) 
print(net.output_shape)
net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
print(net.output_shape)
net = ConvFactory(net, filter_size=1, pad=0, num_filter=64)
print(net.output_shape)
net = ConvFactory(net, filter_size=1, num_filter=1, stride=args.stride)
print(net.output_shape)

output_shape = lasagne.layers.get_output_shape(net)
real_input_shape = (None, input_shape[1], input_shape[2]+2*patch_size, input_shape[3]+2*patch_size)
print("real_input_shape:",real_input_shape,"-> output_shape:",output_shape)

print("network output size should be",(input_shape[2]+2*patch_size)-(patch_size))

if (args.kern == "sq"):
    ef = ((patch_size/args.stride)**2.0)
elif (args.kern == "gaus"):
    ef = 1.0
print("ef", ef)

prediction = lasagne.layers.get_output(net, deterministic=True)
prediction_count = (prediction/ef).sum(axis=(2,3))

classify = theano.function([input_var, input_var_ex], prediction)

train_start_time = time.time()
print(classify(np.zeros((1,channels,framesize,framesize), dtype=theano.config.floatX), [0]).shape)
print(time.time() - train_start_time, "sec")

train_start_time = time.time()
print(classify(np.zeros((1,channels,framesize,framesize), dtype=theano.config.floatX), [0]).shape)
print(time.time() - train_start_time, "sec")