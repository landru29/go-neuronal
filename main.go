package main

import (
	"fmt"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

const nbNeurones = 10
const nbInputs = 2

type training struct {
	inputs  []*mat64.Dense
	targets []*mat64.Dense
}

func main() {

	inputData := [][]float64{
		[]float64{0, 0},
		[]float64{0, 1},
		[]float64{1, 0},
		[]float64{1, 1},
		[]float64{2, 2},
	}

	outputData := [][]float64{
		[]float64{0}, //0+0=0
		[]float64{1}, //0+1=1
		[]float64{1}, //1+0=1
		[]float64{2}, //1+1=2
		[]float64{4}, //1+1=2
	}

	// Prepare the training structure
	var sample training
	for _, data := range inputData {
		sample.inputs = append(sample.inputs, mat64.NewDense(len(data), 1, data))
	}
	for _, data := range outputData {
		sample.targets = append(sample.targets, mat64.NewDense(len(data), 1, data))
	}

	// Initialize W matrix with random numbers between 0 and 1
	W1 := mat64.NewDense(nbNeurones, nbInputs+1, nil)
	r, c := W1.Dims()
	for row := 0; row < r; row++ {
		for col := 0; col < c; col++ {
			W1.Set(row, col, rand.Float64())
		}
	}

	vector := mat64.NewDense(4, 1, []float64{1, 4, 5, 1})

	identity := mat64.NewDense(4, 4, []float64{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1})

	var m mat64.Dense
	m.Mul(identity, vector)

	fmt.Printf("m :\n%v\n\n", mat64.Formatted(vector))
	fmt.Printf("m :\n%v\n\n", mat64.Formatted(identity))
	fmt.Printf("m :\n%v\n\n", mat64.Formatted(&m))
}
