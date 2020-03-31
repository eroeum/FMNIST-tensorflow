from importer import import_csv, split_df
from preprocess import format_df, reshape, to_image_array
from model import Model
from notification import Notifier

def main(proto=False):
    notifier = Notifier("+13013510464")
    notifier.start()

    df = import_csv("/data/fashion-mnist_train.csv")
    train, test = split_df(df)

    train_images, train_labels = format_df(train)
    test_images, test_labels = format_df(test)

    train_images = reshape(train_images)
    test_images = reshape(test_images)

    model = Model(proto=proto)
    model.run(train_images, train_labels)
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    notifier.end()
    notifier.notify_acc(test_acc)

    print("\nTest Accuracy:", test_acc)

    df = import_csv("/data/fashion-mnist_test.csv")
    data = to_image_array(df)
    data = reshape(data)

    predictions = model.predict(data)

    model.write_predictions(predictions)

if __name__ == '__main__':
    main(False)
