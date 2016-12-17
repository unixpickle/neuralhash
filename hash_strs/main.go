package main

import (
	"bytes"
	"fmt"
	"os"

	"github.com/unixpickle/neuralhash"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: hash_strs <network>")
		os.Exit(1)
	}
	net, err := neuralhash.LoadHasher(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load network:", err)
		os.Exit(1)
	}
	for {
		hash := net.Hash(readLine())
		for _, x := range hash {
			if x < 0 {
				fmt.Printf("-")
			} else {
				fmt.Printf("+")
			}
		}
		fmt.Println()
	}
}

func readLine() []byte {
	var res bytes.Buffer
	for {
		var b [1]byte
		if n, err := os.Stdin.Read(b[:]); err != nil {
			os.Exit(0)
		} else if n == 0 {
			continue
		}
		if b[0] == '\r' {
			continue
		} else if b[0] == '\n' {
			break
		}
		res.WriteByte(b[0])
	}
	return res.Bytes()
}
