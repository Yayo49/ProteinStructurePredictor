const neuralNetworkLibrary = require("./neuralNetworkLibrary");
const fs = require("fs");
const MAX_AMINO_ACID_COUNT = 100;
const MAX_SECONDARY_STRUCTURE_COUNT = 100;
const POSSIBLE_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY*".split("");
const POSSIBLE_SECONDARY_STRUCTURES = "CHE".split("");
const INPUT_SIZE = POSSIBLE_AMINO_ACIDS.length * MAX_AMINO_ACID_COUNT;
const OUTPUT_SIZE = POSSIBLE_SECONDARY_STRUCTURES.length * MAX_SECONDARY_STRUCTURE_COUNT;
let datasetCSV = fs.readFileSync("2018-06-06-ss.cleaned.csv");

let lines = [];
let bufferIndex = 0;
lineNumber = 0;
currentLine = ""
while (lineNumber < 1000) {
    let currentChar = datasetCSV.readInt8(bufferIndex);
    if (currentChar == 10) {
	lines.push(currentLine);
	lineNumber++;
	currentLine = ""
    } else {
	currentLine += String.fromCharCode(currentChar);
    }
    bufferIndex++;
}

lines.splice(0, 1);
//console.log(lines);

let aminoAcidStrings = [];
let secondaryStructureStrings = [];

for (let i = 0; i < lines.length; i++) {
    let fields = lines[i].split(",");
    aminoAcidStrings.push(fields[2]);
    secondaryStructureStrings.push(fields[4]);
}

//console.log(aminoAcidStrings);
//console.log(secondaryStructureStrings);

function aminoAcidSequenceToVector(aminoAcidSequence) {
    let aminoAcids = aminoAcidSequence.split("");
    if (aminoAcids.length > MAX_AMINO_ACID_COUNT) {
	console.error("Too many amino acids.");
	return -1;
    }
    let vector = [];
    for (let j = 0; j < aminoAcids.length; j++) {
	let oneHotVector = POSSIBLE_AMINO_ACIDS.map(x => (x == aminoAcids[j]) ? 1 : 0);
	vector = vector.concat(oneHotVector);
    }
    while (vector.length < INPUT_SIZE) {
	vector.push(0);
    }
    return vector;
}

function secondaryStructureSequenceToVector(secondaryStructureSequence) {
    let secondaryStructures = secondaryStructureSequence.split("");
    if (secondaryStructures.length > MAX_SECONDARY_STRUCTURE_COUNT) {
        console.error("Too many secondary structures.");
        return -1;
    }
    let vector = [];
    for (let j = 0; j < secondaryStructures.length; j++) {
        let oneHotVector = POSSIBLE_SECONDARY_STRUCTURES.map(x => (x == secondaryStructures[j]) ? 1 : 0);
        vector = vector.concat(oneHotVector);
    }
    while (vector.length < OUTPUT_SIZE) {
        vector.push(0);
    }
    return vector;
}

/*
function vectorToSecondaryStructureSequence(vector) {
    let secondaryStructureSequence = "";
    for (let i = 0; i < vector.length - (POSSIBLE_SECONDARY_STRUCTURES.length - 1); i += POSSIBLE_SECONDARY_STRUCTURES.length) {
	for (let j = 0; j < POSSIBLE_SECONDARY_STRUCTURES.length; j++) {
	    if (vector[i + j] == 1) {
		secondaryStructureSequence += POSSIBLE_SECONDARY_STRUCTURES[j];
	    }
	}
    }
    return secondaryStructureSequence;
}
*/

//console.log(aminoAcidSequenceToVector("A*").length);
//console.log(secondaryStructureSequenceToVector("CHE"));

//let output = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
//console.log(vectorToSecondaryStructureSequence(output));

function vectorToSecondaryStructureSequence(vector) {
    let secondaryStructureSequence = "";
    for (let i = 0; i < MAX_SECONDARY_STRUCTURE_COUNT; i++) {
	let recordIndex = -1;
	let record = 0;
	for (let j = 0; j < POSSIBLE_SECONDARY_STRUCTURES.length; j++) {
	    let currentComponent = vector[(i * POSSIBLE_SECONDARY_STRUCTURES.length) + j];
	    if (currentComponent > record) {
		recordIndex = j;
		record = currentComponent;
	    }
	}
	if (recordIndex != -1) {
	    secondaryStructureSequence += POSSIBLE_SECONDARY_STRUCTURES[recordIndex];
	}
    }
    return secondaryStructureSequence;
}

//let output = [0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8];
//console.log(vectorToSecondaryStructureSequence(output));

const myNeuralNetwork = new neuralNetworkLibrary.NeuralNetwork(INPUT_SIZE);
myNeuralNetwork.add(100);
myNeuralNetwork.add(OUTPUT_SIZE);
myNeuralNetwork.initialize();

for (let i = 0; i < lines.length; i++) {
    let aminoAcids = aminoAcidStrings[i];
    let secondaryStructures = secondaryStructureStrings[i];
    if ((aminoAcids.length <= MAX_AMINO_ACID_COUNT) && (secondaryStructures.length <= MAX_SECONDARY_STRUCTURE_COUNT)) {
	let inputVector = aminoAcidSequenceToVector(aminoAcidStrings[i]);
	let expectedOutputVector = secondaryStructureSequenceToVector(secondaryStructureStrings[i]);
	myNeuralNetwork.forwardPropagate(inputVector);
	myNeuralNetwork.backPropagate(expectedOutputVector);
	myNeuralNetwork.gradientDescent(0.01);
    }
}
