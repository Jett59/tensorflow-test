import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def generateInputForCharacter(char):
    input = [0.0] * 26
    if char.isalpha():
        input[ord(char) - ord('a')] = 1.0
    return input


def createInput(inputString):
    inputCharacters = []
    for char in inputString:
        inputCharacters.append(char)
    # Fill to 20 characters.
    for i in range(20 - len(inputString)):
        inputCharacters.append(' ')
    # Generate the inputs using generateInputForCharacter on each of the inputCharacters.
    inputs = []
    for char in inputCharacters:
        input = generateInputForCharacter(char)
        for value in input:
            inputs.append(value)
    return inputs


def createOutput(outputString):
    # 'true' - correct or 'false' - not correct
    return [1.0, 0.0] if outputString == 'true' else [0.0, 1.0]


def loadData():
    inputFile = open('inputs.txt', 'r')
    lines = inputFile.readlines()
    inputs = []
    outputs = []
    for line in lines:
        inputWords = line.split()
        inputs.append(createInput(inputWords[0]))
        outputs.append(createOutput(inputWords[1]))
    trainingInputs = np.array(inputs[:int(len(inputs) * 0.8)])
    trainingOutputs = np.array(outputs[:int(len(outputs) * 0.8)])
    testingInputs = np.array(inputs[int(len(inputs) * 0.8):])
    testingOutputs = np.array(outputs[int(len(outputs) * 0.8):])
    return (trainingInputs, trainingOutputs), (testingInputs, testingOutputs)


(train_inputs, train_outputs), (test_inputs,
                                test_outputs) = loadData()


def createModel():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20 * 26, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Start an interractive session where the user types in words and the model figures out if they are correctly spelled.
def interractiveSession(model):
    while True:
        inputString = input("Enter a word: ")
        if inputString == "exit":
            break
        elif inputString == "train":
            model.fit(train_inputs, train_outputs, epochs=1)
            continue
        elif inputString == "save":
            print("Path:")
            path = input()
            model.save(path)
            continue
        elif inputString == "evaluate":
            test_loss, test_acc = model.evaluate(
                test_inputs,  test_outputs, verbose=2)
            print('\nTest accuracy:', test_acc)
            # Find its false-positive and false-negative rate.
            predictions = model.predict(test_inputs)
            falsePositives = 0
            falseNegatives = 0
            for i in range(len(predictions)):
                if np.argmax(predictions[i]) == 1 and np.argmax(test_outputs[i]) == 0:
                    falsePositives += 1
                elif np.argmax(predictions[i]) == 0 and np.argmax(test_outputs[i]) == 1:
                    falseNegatives += 1
            print("False positives: " + str(falsePositives))
            print("False negatives: " + str(falseNegatives))
            continue
        inputArray = createInput(inputString)
        inputArray = np.array([inputArray])
        output = model.predict(inputArray)
        print("Correctly spelled: " + str(output[0][0] > output[0][1]))


print("Load a model or train a new one?")
print("1. Load a model")
print("2. Train a new model")
choice = int(input())
if choice == 1:
    print("Enter the path to the model:")
    modelPath = input()
    model = tf.keras.models.load_model(modelPath)
    interractiveSession(model)
elif choice == 2:
    model = createModel()
    interractiveSession(model)
