package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"strings"
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
	var sampleFile string

	flag.StringVar(&outFile, "output", "out_net", "network file")
	flag.StringVar(&sampleFile, "samples", "", "optional sample word list")
	flag.IntVar(&batchSize, "batch", 100, "batch size")
	flag.IntVar(&hashSize, "hashsize", 10, "hash vector size")
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

	g := &transGradienter{
		G: &Gradienter{
			SeqFunc: &rnn.BlockSeqFunc{B: net.Block},
			Learner: net.Block.(sgd.Learner),
		},
		T: &sgd.Adam{},
	}
	log.Println("Creating samples...")
	s := createSamples(sampleFile)
	log.Println("Training...")
	sgd.SGDMini(g, s, 0.001, batchSize, func(s sgd.SampleSet) bool {
		log.Printf("Cost: %f", g.G.Cost(s).Output()[0])
		return true
	})

	if err := net.Save(outFile); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save:", err)
		os.Exit(1)
	}
}

func createSamples(file string) sgd.SampleSet {
	if file == "" {
		return NewBatch(100000)
	}
	contents, err := ioutil.ReadFile(file)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read samples:", err)
		os.Exit(1)
	}
	strs := strings.Split(string(contents), "\n")
	var res sgd.SliceSampleSet
	for _, x := range strs {
		res = append(res, Sample(x))
	}
	return res
}

type transGradienter struct {
	G *Gradienter
	T sgd.Transformer
}

func (t *transGradienter) Gradient(s sgd.SampleSet) autofunc.Gradient {
	return t.T.Transform(t.G.Gradient(s))
}
