package main

import "github.com/unixpickle/num-analysis/linalg"

type Sample string

func (s Sample) Seq() []linalg.Vector {
	b := []byte(s)
	res := make([]linalg.Vector, len(b))
	for i, x := range b {
		res[i] = make(linalg.Vector, 0x100)
		res[i][int(x)] = 1
	}
	return res
}
