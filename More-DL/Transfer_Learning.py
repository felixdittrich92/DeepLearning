# ...

# Define the CNN
def model_vgg(optimizer, learning_rate, activation_str):
    # Input
    input_img = Input(shape=x_train.shape[1:])

    # VGG base
    conv_base = VGG19(weights='imagenet', include_top=False)(input_img)       

    # Dense Part
    x = GlobalAveragePooling2D()(conv_base)
    x = Activation(activation_str)(x)
    x = Dense(units=num_classes)(x)
    y_pred = Activation("sigmoid")(x)

    # Build the model
    model = Model(inputs=[input_img], outputs=[y_pred])
    opt = optimizer(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.summary()
    return model

# ...