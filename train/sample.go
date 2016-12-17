package main

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
)

// Sample represents a training sample.
type Sample string

// NewBatch creates a batch of related training samples,
// thus forcing the hasher to learn to distinguish between
// very similar samples.
func NewBatch(size int) sgd.SampleSet {
	seed := randomSample()
	res := sgd.SliceSampleSet{seed}
	for len(res) < size {
		seed = seed.smallChange()
		res = append(res, seed)
	}
	return res
}

// Seq produces a sequence of one-hot-vectors based on the
// Sample's bytes.
func (s Sample) Seq() []linalg.Vector {
	b := []byte(s)
	res := make([]linalg.Vector, len(b)+1)
	res[0] = make(linalg.Vector, 0x100)
	for i, x := range b {
		res[i+1] = make(linalg.Vector, 0x100)
		res[i+1][int(x)] = 1
	}
	return res
}

func (s Sample) smallChange() Sample {
	var op int
	if len(s) == 0 {
		op = 0
	} else {
		op = rand.Intn(3)
	}
	switch op {
	case 0: // insert
		idx := rand.Intn(len(s) + 1)
		ch := Sample(byte(rand.Intn(0x100)))
		return s[:idx] + ch + s[idx:]
	case 1: // update
		idx := rand.Intn(len(s))
		ch := Sample(byte(rand.Intn(0x100)))
		return s[:idx] + ch + s[idx+1:]
	case 2: // delete
		idx := rand.Intn(len(s))
		return s[:idx] + s[idx+1:]
	}
	panic("unreachable")
}

func randomSample() Sample {
	size := rand.Intn(30)
	var res Sample
	for i := 0; i < size; i++ {
		res += Sample(byte(rand.Intn(0x100)))
	}
	return res
}
