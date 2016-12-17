package neuralhash

import (
	"io/ioutil"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

// A Hasher uses a neural network to hash data.
type Hasher struct {
	Block rnn.Block
}

// NewHasher creates an untrained hasher with the given
// output hash size.
func NewHasher(hashSize int) *Hasher {
	return &Hasher{
		Block: rnn.StackedBlock{
			rnn.NewLSTM(0x100, 0x100),
			rnn.NewLSTM(0x100, 0x100),
			rnn.NewNetworkBlock(neuralnet.Network{
				neuralnet.NewDenseLayer(0x100, hashSize),
				&neuralnet.HyperbolicTangent{},
			}, 0),
		},
	}
}

// LoadHasher loads a hasher from a file.
func LoadHasher(file string) (*Hasher, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	var h Hasher
	err = serializer.DeserializeAny(data, &h.Block)
	if err != nil {
		return nil, err
	}
	return &h, nil
}

// Save saves the hasher to a file.
func (h *Hasher) Save(path string) error {
	data, err := serializer.SerializeAny(h.Block)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(path, data, 0755)
}

// Hash hashes the byte slice.
func (h *Hasher) Hash(d []byte) linalg.Vector {
	ts := rnn.Runner{Block: h.Block}
	last := ts.StepTime(make(linalg.Vector, 0x100))
	for _, x := range d {
		v := make(linalg.Vector, 0x100)
		v[x] = 1
		last = ts.StepTime(v)
	}
	return last
}
