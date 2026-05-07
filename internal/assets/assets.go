package assets

import _ "embed"

//go:embed model.onnx
var ModelONNX []byte

//go:embed vocab.txt
var VocabTxt []byte
