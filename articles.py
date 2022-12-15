import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

INPUT_INDICES = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:?! \n"

INPUT_CHARACTER_SIZE = len(INPUT_INDICES)

def generateInputsForCharacter(character, outputArray):
    index = INPUT_INDICES.find(character)
    if index == -1:
        raise Exception("Invalid character: " + character + " (" + str(ord(character)) + ")")
    outputArray[index] = 1
    return outputArray


def getCharacterForInput(inputs):
    index = np.argmax(inputs)
    return INPUT_INDICES[index]


def generateInputsForString(str, outputArray):
    index = 0
    stringLength = len(str)
    while index < stringLength:
        character = str[index]
        inputsForCharacter = generateInputsForCharacter(character, outputArray[index])
        index += 1
    return outputArray


INPUT_SIZE = 500
OUTPUT_SIZE = 1


def loadData(path="articles.txt"):
    with open(path) as f:
        data = f.read()
        inputs = np.zeros((len(data) // (INPUT_SIZE + OUTPUT_SIZE), INPUT_SIZE, INPUT_CHARACTER_SIZE))
        outputs = np.zeros((len(data) // (INPUT_SIZE + OUTPUT_SIZE), OUTPUT_SIZE, INPUT_CHARACTER_SIZE))
        i = 0
        for offset in range(0, len(data) - INPUT_SIZE - OUTPUT_SIZE, INPUT_SIZE + OUTPUT_SIZE):
            generateInputsForString(data[offset:offset+INPUT_SIZE], inputs[i])
            generateInputsForString(
                data[offset+INPUT_SIZE:offset+INPUT_SIZE+OUTPUT_SIZE], outputs[i])
            i += 1
        return inputs.reshape((len(inputs), INPUT_SIZE * INPUT_CHARACTER_SIZE)), outputs.reshape((len(outputs), OUTPUT_SIZE * INPUT_CHARACTER_SIZE))


inputs, outputs = loadData()
trainingInputs, trainingOutputs = inputs[:int(
    len(inputs)*0.8)], outputs[:int(len(outputs)*0.8)]
testingInputs, testingOutputs = inputs[int(
    len(inputs)*0.8):], outputs[int(len(outputs)*0.8):]



def createModel():
    global model
    model = models.Sequential()
    model.add(layers.Dense(INPUT_SIZE * INPUT_CHARACTER_SIZE, activation='relu'))
    model.add(layers.Dense(OUTPUT_SIZE *
              INPUT_CHARACTER_SIZE, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def loadModel():
    global model
    model = models.load_model(input("Model path: ").strip())


requestedAction = input("1 to load or 2 to start afresh")
if requestedAction == "1":
    loadModel()
else:
    createModel()


while True:
    inputString = input("Type the beginning: ")
    if inputString == "train":
        model.fit(trainingInputs, trainingOutputs, epochs=1, batch_size=1)
        print(model.evaluate(testingInputs, testingOutputs))
        continue
    if inputString == "save":
        models.save_model(model, input("Model path: ").strip())
        continue
    modelInput = np.zeros((INPUT_SIZE, INPUT_CHARACTER_SIZE))
    generateInputsForString(inputString.rjust(INPUT_SIZE, " "), modelInput)
    tempModelInput = modelInput
    outputCharacterCount = 0
    while outputCharacterCount < 200:
        rawOutput = model(tempModelInput.reshape(
            1, INPUT_SIZE * INPUT_CHARACTER_SIZE), training=False)
        outputCharacter = getCharacterForInput(rawOutput)
        print(outputCharacter, end="", flush=True)
        characterOutputArray = np.zeros(INPUT_CHARACTER_SIZE)
        tempModelInput = np.append(
            tempModelInput[1:], generateInputsForCharacter(outputCharacter, characterOutputArray).reshape(1, INPUT_CHARACTER_SIZE), axis=0)
        outputCharacterCount += 1
    print("")
