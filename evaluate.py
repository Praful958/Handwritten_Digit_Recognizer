from src.evaluation import Evaluator


def main():

    evaluator = Evaluator()

    accuracy, matrix = evaluator.evaluate()


    print(
        f"Test Accuracy: {accuracy*100:.2f}%"
    )


    print("\nConfusion Matrix:")
    print(matrix)



if __name__ == "__main__":
    main()