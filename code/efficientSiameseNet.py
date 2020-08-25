import tensorflow as tf

def constructCnnBackbone(imageSize = 224):
    netInput = tf.keras.Input(shape=(imageSize, imageSize, 3), name="backboneInput")
    backbone = tf.keras.applications.EfficientNetB0(        
        weights='imagenet',
        include_top=False,
        input_shape=(imageSize, imageSize, 3),
        # it should have exactly 3 inputs channels,
        pooling=None)  # Tx7x7x1280 in case of None pooling and image side size of 224
    converted = tf.keras.applications.efficientnet.preprocess_input(netInput)
    print("converted cnn backbone input shape {0}".format(converted.shape))
    #print("Backbone")
    #print(backbone.summary())

    result = backbone(converted)
    return tf.keras.Model(name="Backbone", inputs=netInput, outputs=result), backbone

def constructFeatureExtractor(backboneModel, seriesLen, l2regAlpha, DORate, seed, imageSize = 224):
    netInput = tf.keras.Input(shape=(seriesLen, imageSize, imageSize, 3), name="featureExtractorInput")
    backboneApplied = tf.keras.layers.TimeDistributed(backboneModel,name="backbone")(netInput) # 7x7x1280 (for image size 224) = 62,720

    backboneOutChannelsCount = 1280

    # we will do 2D convolution until the image become 1x1
    cnn1out = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(backboneOutChannelsCount // 2, kernel_size=1,strides=(1,1),padding='valid',activation='selu'),name="postBackboneConv2D")(backboneApplied)
    cnn1DoOut = tf.keras.layers.AlphaDropout(DORate, noise_shape=(seriesLen,1,1,backboneOutChannelsCount // 2),seed=seed+2334)(cnn1out) # 7 x 7 x 640
    cnn2out = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(backboneOutChannelsCount // 4, kernel_size=3,strides=(2,2),padding='valid',activation='selu'),name="postBackboneConv2D_2")(cnn1DoOut)
    cnn2DoOut = tf.keras.layers.AlphaDropout(DORate, noise_shape=(seriesLen,1,1,backboneOutChannelsCount // 4),seed=seed+34632)(cnn2out) # 3 x 3 x 320
    cnn3out = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(backboneOutChannelsCount // 4, kernel_size=3,strides=(1,1),padding='valid',activation='selu'),name="postBackboneConv2D_3")(cnn2DoOut)
    cnn3DoOut = tf.keras.layers.AlphaDropout(DORate, noise_shape=(seriesLen,1,1,backboneOutChannelsCount // 4),seed=seed+2346)(cnn3out) # 1 x 1 x 320
    cnnFinal = tf.keras.layers.Reshape((seriesLen, backboneOutChannelsCount // 4))(cnn3DoOut)
    fc1out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(backboneOutChannelsCount // 8, activation="selu", kernel_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha)),name="fc1")(cnnFinal)
    fc1DoOut =  tf.keras.layers.AlphaDropout(DORate, noise_shape=(seriesLen,backboneOutChannelsCount // 8),seed=seed+245334)(fc1out)
    rnnOut = \
        tf.keras.layers.GRU(
            backboneOutChannelsCount // 16, dropout=DORate,
            kernel_regularizer = tf.keras.regularizers.L1L2(l2=l2regAlpha),
            recurrent_regularizer=tf.keras.regularizers.L1L2(l2=l2regAlpha),            
            return_sequences=False)(fc1DoOut)
    result = tf.keras.Model(name="FeatureExtractor", inputs=netInput, outputs=rnnOut)
    print("Feature extractor")
    print(result.summary())
    return result


def constructSiameseTripletModel(seriesLen, l2regAlpha, DORate, imageSize = 224, optimizationMargin = 2.0):
    anchorInput = tf.keras.Input(shape=(seriesLen, imageSize, imageSize, 3), name="anchorInput")
    poitiveInput = tf.keras.Input(shape=(seriesLen, imageSize, imageSize, 3), name="positiveInput")
    negativeInput = tf.keras.Input(shape=(seriesLen, imageSize, imageSize, 3), name="negativeInput")

    backbone,backboneCore = constructCnnBackbone(imageSize)
    featureExtractor = constructFeatureExtractor(backbone, seriesLen, l2regAlpha, DORate, imageSize)
    anchorFeatures = featureExtractor(anchorInput) # B x features
    positiveFeatures = featureExtractor(poitiveInput)
    negativeFeatures = featureExtractor(negativeInput)

    result = tf.keras.Model(name="SiameseTripletModel", inputs=[anchorInput, poitiveInput, negativeInput], outputs=[anchorFeatures, positiveFeatures, negativeFeatures])

    # adding unsupervised loss
    posSim = tf.keras.losses.cosine_similarity(anchorFeatures, positiveFeatures, axis=-1) # -1.0 is perferct alignment
    negSim = tf.keras.losses.cosine_similarity(anchorFeatures, negativeFeatures, axis=-1)
    print("posSim shape {0}".format(posSim.shape))
    loss = tf.reduce_mean(tf.maximum(optimizationMargin + posSim - negSim, 0))

    result.add_loss(loss)
    
    return result, backbone, featureExtractor