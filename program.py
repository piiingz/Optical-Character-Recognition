from character_detection import run_character_detection
from preprocessing import run_preprocess_train_test


def main():
    # Should the networks be trained from scratch?
    # Note: The SVM-model does not get saved, but the CNN comes with a
    # pretrained model due to long training times with our hyperparameters
    train_svm = True
    train_cnn = False

    # Run the networks, training and testing - Printing and plotting all relevant information
    run_preprocess_train_test(train_svm, train_cnn)
    run_character_detection()


if __name__ == "__main__":
    main()
