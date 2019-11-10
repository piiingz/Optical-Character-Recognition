from sklearn.svm import SVC


def svc_model(images, labels, test_image):
    images = images.reshape([images.shape[0], 400])
    test_image = test_image.reshape([test_image.shape[0], 400])
    svc = SVC(gamma="scale", kernel='rbf')
    svc.fit(images, labels)
    prediction = svc.predict(test_image)
    print("Done predicting")

    return prediction
