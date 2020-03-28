from importer import import_csv, split_df
from preprocess import format_df, reshape
from model import Model

DATA_SPLIT = 0.8
BATCH_SIZE = 64
N_EPOCHS   = 20

def main():
    df = import_csv("/data/fashion-mnist_train.csv")
    train, test = split_df(df)

    train_images, train_labels = format_df(train)
    test_images, test_labels = format_df(test)

    train_images = reshape(train_images)
    test_images = reshape(test_images)

    model = Model()
    model.run(train_images, train_labels)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("\nTest Accuracy:", test_acc)

if __name__ == '__main__':
    main()
