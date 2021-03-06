package main

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
)

// A Gradienter computes gradients for the batch-oriented
// loss function.
type Gradienter struct {
	SeqFunc seqfunc.Func
	Learner sgd.Learner
}

// Gradient computes the gradient for the batch.
func (g *Gradienter) Gradient(s sgd.SampleSet) autofunc.Gradient {
	grad := autofunc.NewGradient(g.Learner.Parameters())
	g.Cost(s).PropagateGradient([]float64{1}, grad)
	return grad
}

// Cost computes the cost for the batch.
func (g *Gradienter) Cost(s sgd.SampleSet) autofunc.Result {
	seqs := make([][]linalg.Vector, s.Len())
	for i := range seqs {
		seqs[i] = s.GetSample(i).(Sample).Seq()
	}
	in := seqfunc.ConstResult(seqs)
	out := seqfunc.ConcatLast(g.SeqFunc.ApplySeqs(in))
	return autofunc.Pool(out, func(out autofunc.Result) autofunc.Result {
		comps := len(out.Output()) / s.Len()
		cov := covarianceMatrix(out, s.Len())
		mask := &autofunc.Variable{Vector: make(linalg.Vector, comps*comps)}
		covWeights := 1 / float64(comps-1)
		for i := range mask.Vector {
			if i%comps == i/comps {
				mask.Vector[i] = -1
			} else if cov.Output()[i] < 0 {
				mask.Vector[i] = -covWeights
			} else {
				mask.Vector[i] = covWeights
			}
		}
		return autofunc.SumAll(autofunc.Mul(mask, cov))
	})
}

func covarianceMatrix(result autofunc.Result, n int) autofunc.Result {
	centered := subtractMeans(result, n)
	cols := len(centered.Output()) / n
	trans := autofunc.Transpose(centered, n, cols)
	return autofunc.Pool(trans, func(trans autofunc.Result) autofunc.Result {
		res := autofunc.MatMulVecs(trans, cols, n, trans)
		return autofunc.Scale(res, 1/float64(n))
	})
}

func subtractMeans(result autofunc.Result, n int) autofunc.Result {
	m := autofunc.Scale(computeMean(result, n), -1)
	return autofunc.Pool(m, func(m autofunc.Result) autofunc.Result {
		rep := autofunc.Repeat(m, n)
		return autofunc.Add(result, rep)
	})
}

func computeMean(result autofunc.Result, n int) autofunc.Result {
	split := autofunc.Split(n, result)
	sum := split[0]
	for _, x := range split[1:] {
		sum = autofunc.Add(sum, x)
	}
	return autofunc.Scale(sum, 1/float64(n))
}
