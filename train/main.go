package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/neuralhash"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var hashSize int
	var batchSize int
	var outFile string
	var stepSize float64

	flag.StringVar(&outFile, "output", "out_net", "network file")
	flag.IntVar(&batchSize, "batch", 100, "batch size")
	flag.IntVar(&hashSize, "hashsize", 32, "hash vector size")
	flag.Float64Var(&stepSize, "step", 0.001, "SGD step size")

	flag.Parse()

	net, err := neuralhash.LoadHasher(outFile)
	if os.IsNotExist(err) {
		log.Println("Creating new network")
		net = neuralhash.NewHasher(hashSize)
	} else if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load network:", err)
		os.Exit(1)
	}

	ig := &ignoreGradienter{
		G: &Gradienter{
			SeqFunc: &rnn.BlockSeqFunc{B: net.Block},
			Learner: net.Block.(sgd.Learner),
		},
		Trans: &sgd.Adam{},
	}
	fakeSamples := sgd.SliceSampleSet{nil}
	sgd.SGDMini(ig, fakeSamples, 0.001, 1, func(_ sgd.SampleSet) bool {
		ig.Batch = NewBatch(batchSize)
		log.Printf("Cost: %f", ig.G.Cost(ig.Batch).Output()[0])
		return true
	})

	if err := net.Save(outFile); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save:", err)
		os.Exit(1)
	}
}

type ignoreGradienter struct {
	G     *Gradienter
	Batch sgd.SampleSet
	Trans sgd.Transformer
}

func (i *ignoreGradienter) Gradient(s sgd.SampleSet) autofunc.Gradient {
	return i.Trans.Transform(i.G.Gradient(i.Batch))
}
