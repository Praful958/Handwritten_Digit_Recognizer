import matplotlib.pyplot as plt

from src.evaluation import Evaluator


def main():

    evaluator = Evaluator()

    accuracy, matrix = evaluator.evaluate()

    print(f"✅ Test Accuracy: {accuracy:.2f}%")

    plt.figure(figsize=(8, 6))

    plt.imshow(matrix)

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.colorbar()

    plt.savefig(
        "outputs/confusion_matrix.png"
    )

    plt.close()

    print("✅ Confusion Matrix Saved")


if __name__ == "__main__":
    main()