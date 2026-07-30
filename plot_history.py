import json
import matplotlib.pyplot as plt


# Load History

with open("outputs/history.json", "r") as f:
    history = json.load(f)


loss = history["loss"]
accuracy = history["accuracy"]


# Loss Graph

plt.figure(figsize=(8,5))

plt.plot(loss)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid()

plt.savefig("outputs/loss_graph.png")

plt.close()



# Accuracy Graph

plt.figure(figsize=(8,5))

plt.plot(accuracy)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy (%)")

plt.grid()

plt.savefig("outputs/accuracy_graph.png")

plt.close()


print("✅ Graphs Generated Successfully")