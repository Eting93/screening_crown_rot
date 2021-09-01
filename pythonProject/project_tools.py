def som(map_row, map_col, input_data, labels, flag_pbc, step=0.01, iteration=1000):
    """
    Conduct SOM clustering.
    :param map_row:
    :param map_col:
    :param input_data:
    :param labels:
    :param flag_pbc:
    :param step:
    :param iteration:
    :return:
    """
    import numpy as np
    import SimpSOM as sps

    print('Conductiong SOM')
    # Build a network 20x20 with a weights format taken from the ref and activate Periodic Boundary Conditions.
    net = sps.somNet(map_row, map_col, input_data, PBC=flag_pbc)

    # Train the network for 10000 epochs and with initial learning rate of 0.01.
    net.train(step, iteration)

    # Save the weights to file
    net.save('som_weights')
    # weight = np.load('som_weights.npy')

    net.nodes_graph(colnum=0)  # Plot a figure of node feature (column 0) and save the figure in the PWD
    net.diff_graph()  # Plot a figure of weight difference and save it in the PWD

    # Project the datapoints on the new 2D network map.
    net.project(input_data, labels=labels)  # Project the labels to the weight different figure and save it.

    # Cluster the datapoints according to the Quality Threshold algorithm.
    net.cluster(input_data, type='qthresh')  # Do clustering and save a figure


def k_means(input_data, random_state, n_clusters, labels):
    """
    :param input_data:
    :param random_state:
    :param n_clusters:
    :return:
    """
    from sklearn.cluster import KMeans
    from matplotlib import pyplot as plt

    print("Conducting k-means")

    # Create a KMeans object
    k_means = KMeans(random_state=random_state, n_clusters=n_clusters)
    print('k-means parameter:')
    print(k_means.get_params())

    # Fit
    k_means.fit(input_data)
    print('K-means cluster centres:')
    print(k_means.cluster_centers_)
    print('K-means inertia:')
    print(k_means.inertia_)

    # Predict
    predictions = k_means.predict(input_data)
    print('predictions')
    print(predictions)

    # Cluster analysis
    plt.figure()
    plt.scatter(input_data[:, 0], input_data[:, 1], c=predictions, s=30, cmap='cividis')
    plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], c='blue', s=200, alpha=0.5)
    plt.title('K-means classification', fontsize=12)

    for i in range(0, labels.shape[0]):
        plt.text(input_data[i, 0], input_data[i, 1], labels[i])
    plt.pause(2)


def nn_multiclass_logistic_regression(input_data, labels, num_hidden, num_output, batch_size, learning_rate=0.01, epochs=10,
                      smoothing_constant=0.01, flag_shuffle=True):
    import mxnet as mx
    from mxnet import nd, autograd, gluon
    import numpy as np

    print('Conducting nn_multiclass_logistic_regression.')

    ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cup()
    data_ctx = ctx
    model_ctx = ctx

    # Training data
    input_data = mx.nd.array(input_data) # 2D
    labels = mx.nd.array(labels) # 1D

    train_data = mx.gluon.data.DataLoader(gluon.data.ArrayDataset(input_data, labels),
                                          batch_size=batch_size, shuffle=flag_shuffle)

    # Define net
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(num_hidden, activation='relu'))
        net.add(gluon.nn.Dense(num_output))

    # Parameter initialization
    net.collect_params().initialize(mx.init.Normal(sigma=0.1), ctx=model_ctx)

    # Softmax corss-entropy loss
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    # Optimizer
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})

    # Evaluation metric
    def evaluate_accuracy(data_iterator, net):
        acc = mx.metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]

    # Training loop
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0]) # ?
            cumulative_loss += nd.sum(loss).asscalar()

        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s" %
              (e, cumulative_loss / input_data.shape[0], train_accuracy))


def rgb_stem_statistics(data_path, data_name, flag_figure=False):
    """
    Calculate the statistics of a RGB image of a stem.
    :param data_path:
    :param data_name:
    :param flag_fig:
    :return: A dictionary containing the statistics.
    """

    import numpy as np
    from matplotlib import pyplot as plt
    from skimage import img_as_float
    from skimage.color import rgb2hsv, rgb2lab

    # RGB
    rgb = plt.imread(data_path + '/' + data_name)
    rgb = img_as_float(rgb)
    rgb[rgb == [1, 1, 1]] = 0

    ave_red = np.mean(rgb[:, :, 0])
    ave_green = np.mean(rgb[:, :, 1])
    ave_blue = np.mean(rgb[:, :, 2])

    if flag_figure:
        f_rgb, a_f_rgb = plt.subplots(2, 2)
        a_f_rgb[0, 0].imshow(rgb)
        a_f_rgb[0, 0].set_title('RGB imag')
        a_f_rgb[0, 1].imshow(rgb[:, :, 0], cmap='jet')
        a_f_rgb[0, 1].set_title('Red')
        a_f_rgb[1, 0].imshow(rgb[:, :, 1], cmap='jet')
        a_f_rgb[1, 0].set_title('Green')
        a_f_rgb[1, 1].imshow(rgb[:, :, 2], cmap='jet')
        a_f_rgb[1, 1].set_title('Blue')
        plt.pause(2)

    # HSV
    hsv = rgb2hsv(rgb)
    ave_h = np.mean(hsv[:, :, 0])
    ave_s = np.mean(hsv[:, :, 1])
    ave_v = np.mean(hsv[:, :, 2])

    if flag_figure:
        f_hsv, a_f_hsv = plt.subplots(2, 2)
        a_f_hsv[0, 0].imshow(hsv)
        a_f_hsv[0, 0].set_title('HSV imag')
        a_f_hsv[0, 1].imshow(hsv[:, :, 0], cmap='jet')
        a_f_hsv[0, 1].set_title('Hue')
        a_f_hsv[1, 0].imshow(hsv[:, :, 1], cmap='jet')
        a_f_hsv[1, 0].set_title('Saturation')
        a_f_hsv[1, 1].imshow(hsv[:, :, 2], cmap='jet')
        a_f_hsv[1, 1].set_title('Value')
        plt.pause(2)

    # The normalised histogram of a in Lab
    hist_h, _ = np.histogram(hsv[:, :, 0].reshape(rgb.shape[0] * rgb.shape[1]), bins=10, density=True)
    if flag_figure:
        plt.figure()
        plt.plot(hist_h)

    # lab
    lab = rgb2lab(rgb)
    ave_l = np.mean(lab[:, :, 0])
    ave_a = np.mean(lab[:, :, 1])
    ave_b = np.mean(lab[:, :, 2])

    if flag_figure:
        f_lab, a_f_lab = plt.subplots(2, 2)
        a_f_lab[0, 0].imshow(lab)
        a_f_lab[0, 0].set_title('lab imag')
        a_f_lab[0, 1].imshow(lab[:, :, 0], cmap='jet')
        a_f_lab[0, 1].set_title('Lightness')
        a_f_lab[1, 0].imshow(lab[:, :, 1], cmap='jet')
        a_f_lab[1, 0].set_title('a')
        a_f_lab[1, 1].imshow(lab[:, :, 2], cmap='jet')
        a_f_lab[1, 1].set_title('b')

    # The normalised histogram of a in Lab
    hist_a, _ = np.histogram(lab[:, :, 0].reshape(rgb.shape[0] * rgb.shape[1]), bins=10, density=True)
    if flag_figure:
        plt.figure()
        plt.plot(hist_a)

    dict = {'average red': ave_red,
            'average green': ave_green,
            'average blue': ave_blue,
            'average hue': ave_h,
            'average saturation': ave_s,
            'average value': ave_v,
            'average lightness': ave_l,
            'average a': ave_a,
            'average b': ave_b,
            'hist_a': hist_a,
            'hist_h': hist_h}

    return dict






