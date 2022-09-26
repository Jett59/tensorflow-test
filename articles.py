import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

INPUT_CHARACTER_SIZE = 128


def generateInputsForCharacter(character, outputArray):
    if ord(character) != 0:
        outputArray[ord(character)] = 1.0
    return outputArray


def getCharacterForInput(inputs):
    return chr(np.argmax(inputs))


def generateInputsForString(str, outputArray):
    index = 0
    stringLength = len(str)
    while index < stringLength:
        character = str[index]
        inputsForCharacter = generateInputsForCharacter(character, outputArray[index])
        index += 1
    return outputArray


INPUT_SIZE = 100
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
    model.add(layers.Dense(INPUT_SIZE * INPUT_CHARACTER_SIZE / 4, activation='relu'))
    model.add(layers.Dense(OUTPUT_SIZE *
              INPUT_CHARACTER_SIZE, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def loadModel():
    global model
    model = models.load_model(input("Model path: "))


requestedAction = input("1 to load or 2 to start afresh")
if requestedAction == "1":
    loadModel()
else:
    createModel()


while True:
    inputString = input("Type the beginning: ")
    if inputString == "train":
        print(trainingInputs.shape, trainingOutputs.shape)
        model.fit(trainingInputs, trainingOutputs, epochs=1, batch_size=256)
        print(model.evaluate(testingInputs, testingOutputs))
        continue
    if inputString == "save":
        model.save(input("Model path: "))
        continue
    modelInput = np.zeros((INPUT_SIZE, INPUT_CHARACTER_SIZE))
    generateInputsForString(inputString.rjust(INPUT_SIZE, "\0"), modelInput)
    tempModelInput = modelInput
    outputCharacterCount = 0
    while outputCharacterCount < 500:
        rawOutput = model(tempModelInput.reshape(
            1, INPUT_SIZE * INPUT_CHARACTER_SIZE), training=False)
        outputCharacter = getCharacterForInput(rawOutput)
        if outputCharacter == '\0':
            break
        print(outputCharacter, end="", flush=True)
        characterOutputArray = np.zeros(INPUT_CHARACTER_SIZE)
        tempModelInput = np.append(
            tempModelInput[1:], generateInputsForCharacter(outputCharacter, characterOutputArray).reshape(1, INPUT_CHARACTER_SIZE), axis=0)
        outputCharacterCount += 1
    print("")
