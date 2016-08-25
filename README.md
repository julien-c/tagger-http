## tagger-http

Basic HTTP server for [Tagger](https://github.com/glample/tagger), implementation of a Named Entity Recognizer that obtains state-of-the-art performance in NER on the 4 CoNLL datasets (English, Spanish, German and Dutch) without resorting to any language-specific knowledge or resources such as gazetteers.

The model is only loaded once when the server starts. Requests are then pretty quick, around `~150ms` on my machine.

## Response format

```
GET /?q=Barack Obama is president of the USA
```

Output:

```json
{
	"text": "Barack Obama is president of the USA",
	"ranges": [  
		[ 0, 1, "PER" ],
		[ 6, 6, "ORG" ]
	],
	"words": [  
		"Barack",
		"Obama",
		"is",
		"president",
		"of",
		"the",
		"USA"
	]
}
```

## Sources

Hat/tip [@glample](https://github.com/glample) and [@a455bcd9](https://github.com/a455bcd9)
