import numpy as np
import math

class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0

class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    #Because why not
    def addError(self, err):
        self.error = self.error + err

    #For Squishification
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    #The Derivative of the Sigmoid Function
    def dSigmoid(self, x):
        return x * (1.0 - x)

    #Once again, why not
    def setError(self, err):
        self.error = err

    #Im a normal person I swear
    def setOutput(self, output):
        self.output = output

    #Well, I do have some parent issues
    def getOutput(self):
        return self.output

    #Gets total output from connect neurons
    def feedForword(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            try:
                sumOutput = float(sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight)
            except:
                print('Feed Forword error:')
                print(sumOutput)
        self.output = self.sigmoid(sumOutput)

    #Find the gradient of the function, and then adjusts all of the weights and such accordingly
    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output)
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight
            dendron.weight = dendron.weight + dendron.dWeight
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)
        self.error = 0


class Network:
    def __init__(self, topology):
        self.layers = []
        #create all of the neurons
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    #pretty straight forward
    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    #well im just making these for fun now
    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword()

    #making the network do stuff
    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    #math that need'nt be understood
    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    #our output
    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()
        return output

    #we finally get some stuff
    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            if (o > 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()
        return output
    
#Function to make things easier ilayers = input layers, hlayers = hidden layers, olayers = output layers
def train(ilayers, hlayers, olayers, inputs, outputs, finalerr = 0.05):
    
    #Setup the layers of the network
    topology = []
    topology.append(ilayers)
    topology.append(hlayers)
    topology.append(olayers)
    net = Network(topology)
    
    #stuff to make network go faster or slower
    
    Neuron.eta = 0.09
    Neuron.alpha = 0.015
    while True:
        #can't have an error at the start
        err = 0
        
        #Do the stuff
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForword()
            net.backPropagate(outputs[i])
            err = err + net.getError(outputs[i])
        print ("error: ", err)
        
        #If breakpoint is reached, stop the training!
        if err < finalerr:
            print('Training has finished!')
            testOsave = str(input('Test NN (t) or Save NN (s)): '))
            if testOsave == 't':
                inputs = []
                for i in range(ilayers):
                    currentNumber = i+1
                    currentNumberStr = str(currentNumber)
                    inputs.append(int(input('Number' + currentNumberStr + ': ')))
                net.setInput(inputs)
                print('now testing: ')
                print(inputs)
                #net.setInput([1,0])
                net.feedForword()
                print('the network says: ')
                print (net.getThResults())
                testRsave = str(input('Save NN (s) or Delete NN (d): '))
                if testRsave == 'd':
                    break
                else:
                    pickle.dump(net, open( "nn.p", "wb" ))
            else:
                pickle.dump(net, open( "nn.p", "wb" ))
                
#ignore all of this
def run(filename, inputs):
    net = pickle.load(open(filename, "rb"))
    i = []
    tt = 3
    for iter in range(inputs):
        a = int(sys.argv[tt])
        i.append(a)
        tt += 1
    net.setInput(i)
    net.feedForword()
    print (net.getThResults())


inputs = [[1,0,1,1],[0,1,0,0],[1,0,0,1],[0,1,1,0],[1,0,0,0]]
outputs = [[0],[1],[0],[1],[0]]
train(4,2,1,inputs,outputs,0.1)
