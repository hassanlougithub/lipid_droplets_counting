

target_var = T.tensor4('target')
lr = theano.shared(np.array(args.lr, dtype=theano.config.floatX))

#Mean Absolute Error is computed between each count of the count map
l1_loss = T.abs_(prediction - target_var[input_var_ex])

#Mean Absolute Error is computed for the overall image prediction
prediction_count2 =(prediction/ef).sum(axis=(2,3))
mae_loss = T.abs_(prediction_count2 - (target_var[input_var_ex]/ef).sum(axis=(2,3))) 

loss = l1_loss.mean()

params = lasagne.layers.get_all_params(net, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=lr)

train_fn = theano.function([input_var_ex], [loss,mae_loss], updates=updates,
                         givens={input_var:np_dataset_x_train, target_var:np_dataset_y_train})

print ("DONE compiling theano functons")
best_valid_err = 99999999
best_test_err = 99999999
epoch = 0
batch_size = 2

print ("batch_size", batch_size)
print ("lr", lr.eval())

datasetlength = len(np_dataset_x_train)
print ("datasetlength",datasetlength)

for epoch in range(epoch, 1000):
    start_time = time.time()

    epoch_err_pix = []
    epoch_err_pred = []
    todo = range(datasetlength)    
    
    for i in range(0,datasetlength, batch_size):
        ex = todo[i:i+batch_size]

        train_start_time = time.time()
        err_pix,err_pred = train_fn(ex)
        train_elapsed_time = time.time() - train_start_time

        epoch_err_pix.append(err_pix)
        epoch_err_pred.append(err_pred)

    valid_pix_err, valid_err = test_perf(np_dataset_x_valid, np_dataset_y_valid, np_dataset_c_valid)

    # a threshold is used to reduce processing when we are far from the goal
    if (valid_err < 11 and valid_err < best_valid_err):
        best_valid_err = valid_err
        best_test_err = test_perf(np_dataset_x_test, np_dataset_y_test,np_dataset_c_test)
        print ("OOO best test (err_pix, err_pred)", best_test_err, ",epoch",epoch)
        os.chdir('/valohai/outputs/result/network-temp')   
        save_network(net,"best_valid_err")
       


    elapsed_time = time.time() - start_time
    err = np.mean(epoch_err_pix)
    acc = np.mean(np.concatenate(epoch_err_pred))
    
    if epoch % 5 == 0:
        print ("#" + str(epoch) + "#(err_pix:" + str(np.around(err,3)) + ",err_pred:" +  str(np.around(acc,3)) + "),valid(err_pix:" + str(np.around(valid_pix_err,3)) + ",err_pred:" + str(np.around(valid_err,3)) +"),(time:" + str(np.around(elapsed_time,3)) + "sec)")

    
    

print ("#####", "best_test_acc", best_test_err, args)
print("Done")