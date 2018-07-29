class Layer {
	constructor(numberOfNeurons) {
		this.numberOfNeurons = numberOfNeurons;
		this.biases = [];
	}
  initialize() {
    for (let i = 0; i < this.numberOfNeurons; i++) {
			this.biases.push((Math.random() * 2) - 1);
		}
  }
	sigmoid(x) {
		return 1 / (1 + Math.exp(x * -1));
	}
	sigmoidPrime(x) {
		let y = this.sigmoid(x);
		return y * (1 - y);
	}
	forwardPropagate(input) {
		this.input = input;
		this.output = [];
		for (let i = 0; i < this.numberOfNeurons; i++) {
			this.output.push(this.sigmoid(input[i] - this.biases[i]));
		}
		return this.output;
	}
	backPropagate(input) {
		let output = [];
		for (let i = 0; i < this.numberOfNeurons; i++) {
			output[i] = input[i] * this.sigmoidPrime(this.input[i]);
		}
    this.error = output;
		return output;
	}
  gradientDescent(learningRate) {
    for (let i = 0; i < this.biases.length; i++) {
      this.biases[i] -= this.error[i] * learningRate;
    }
  }
}

class WeightMatrix {
	constructor(inputLayerLength, outputLayerLength) {
		this.inputLayerLength = inputLayerLength;
		this.outputLayerLength = outputLayerLength;
		this.weights = [];
	}
  initialize() {
    for (let y = 0; y < this.inputLayerLength; y++) {
			this.weights[y] = [];
			for (let x = 0; x < this.outputLayerLength; x++) {
				this.weights[y][x] = (Math.random() * 2) - 1;
			}
		}
  }
	forwardPropagate(input) {
		this.input = input;
		this.output = [];
		for (let x = 0; x < this.outputLayerLength; x++) {
			this.output.push(0);
			for (let y = 0; y < this.inputLayerLength; y++) {
				this.output[x] += input[y] * this.weights[y][x];
			}
		}
		return this.output;
  }
  backPropagate(input) {
    let output = [];
    for (let y = 0; y < this.inputLayerLength; y++) {
      output[y] = 0;
      for (let x = 0; x < this.outputLayerLength; x++) {
        output[y] += input[x] * this.weights[y][x];
      }
    }
    return output;
  }
  gradientDescent(learningRate, outputOfInputLayer, errorOfOutputLayer) {
    let weightsPrime = [];
    for (let y = 0; y < this.inputLayerLength; y++) {
      weightsPrime.push([]);
      for (let x = 0; x < this.outputLayerLength; x++) {
        weightsPrime[y][x] = outputOfInputLayer[y] * errorOfOutputLayer[x];
        this.weights[y][x] += weightsPrime[y][x] * learningRate;
      }
    }
  }
}

class NeuralNetwork {
	constructor(inputLayerLength) {
    this.architecture = [inputLayerLength];
    this.layers = [new Layer(inputLayerLength)];
    this.weightMatrices = [];
  }
  add(lengthOfNewLayer) {
    this.weightMatrices.push(new WeightMatrix(this.layers[this.layers.length - 1].numberOfNeurons, lengthOfNewLayer));
    this.layers.push(new Layer(lengthOfNewLayer));
    this.architecture.push(lengthOfNewLayer);
  }
  initialize() {
    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].initialize();
    }
    for (let i = 0; i < this.weightMatrices.length; i++) {
      this.weightMatrices[i].initialize();
    }
  }
  forwardPropagate(input) {
    let currentOutput = this.layers[0].forwardPropagate(input);
    for (let i = 0; i < this.weightMatrices.length; i++) {
      currentOutput = this.layers[i + 1].forwardPropagate(this.weightMatrices[i].forwardPropagate(currentOutput));
    }
    return currentOutput;
  }
  backPropagate(expectedOutput) {
    let currentOutput = [];
    for (let i = 0; i < expectedOutput.length; i++) {
      currentOutput[i] = expectedOutput[i] - this.layers[this.layers.length - 1].output[i];
    }
    currentOutput = this.layers[this.layers.length - 1].backPropagate(currentOutput);
    for (let i = this.weightMatrices.length - 1; i >= 0; i--) {
      currentOutput = this.weightMatrices[i].backPropagate(currentOutput);
      currentOutput = this.layers[i].backPropagate(currentOutput);
    }
  }
  gradientDescent(learningRate) {
    for (let i = 0; i < this.layers.length; i++) {
      this.layers[i].gradientDescent(learningRate);
    }
    for (let i = 0; i < this.weightMatrices.length; i++) {
      this.weightMatrices[i].gradientDescent(learningRate, this.layers[i].output, this.layers[i + 1].error);
    }
  }
}

/*
function breedVectors(vectors, mutationRate) {
  let offspring = [];
  for (let i = 0; i < vectors[0].length; i++) {
    if (Math.random() <= mutationRate) {
      offspring[i] = (Math.random() * 2) - 1;
    } else {
      offspring[i] = vectors[Math.floor(Math.random() * vectors.length)][i];
    }
  }
  return offspring;
}

function breedMatrices(matrices, mutationRate) {
  let offspring = [];
  for (let y = 0; y < matrices[0].length; y++) {
    offspring[y] = [];
    for (let x = 0; x < matrices[0][0].length; x++) {
      if (Math.random() <= mutationRate) {
        offspring[y][x] = (Math.random() * 2) - 1;
      } else {
        offspring[y][x] = matrices[Math.floor(Math.random() * matrices.length)][y][x];
      }
    }
  }
  return offspring;
}
*/

function breedNeuralNetworks(neuralNetworks, mutationRate) {
  let architecture = neuralNetworks[0].architecture;
  let offspring = new NeuralNetwork(architecture[0]);
  for (let i = 1; i < architecture.length; i++) {
    offspring.add(architecture[i]);
  }
  for (let i = 0; i < architecture.length; i++) {
    for (let j = 0; j < architecture[i]; j++) {
      if (Math.random() <= mutationRate) {
        offspring.layers[i].biases[j] = (Math.random() * 2) - 1;
      } else {
       offspring.layers[i].biases[j] = neuralNetworks[Math.floor(Math.random() * neuralNetworks.length)].layers[i].biases[j];
      }
    }
  }
  for (let i = 0; i < architecture.length - 1; i++) {
    for (let y = 0; y < architecture[i]; y++) {
      offspring.weightMatrices[i].weights[y] = [];
      for (let x = 0; x < architecture[i + 1]; x++) {
        if (Math.random() <= mutationRate) {
          offspring.weightMatrices[i].weights[y][x] = (Math.random() * 2) - 1;
        } else {
          offspring.weightMatrices[i].weights[y][x] = neuralNetworks[Math.floor(Math.random() * neuralNetworks.length)].weightMatrices[i].weights[y][x];
        }
      }
    }
  }
  return offspring;
}

module.exports = {Layer: Layer,
WeightMatrix: WeightMatrix,
NeuralNetwork: NeuralNetwork,
breedNeuralNetworks: breedNeuralNetworks};
