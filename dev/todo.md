- [x] run inference, save to output.txt
  - `./build/bin/llama-simple -m models/gemma/gemma-1.1-7b-it.Q4_K_M.gguf -n 100 -p "Tell me about the history of artificial intelligence" >> output.txt`
  New way:
  ```
  ./build/bin/llama-run --ngl 999 models/gemma/gemma-1.1-7b-it.Q4_K_M.gguf Hello World  > output.txt 
  ```

- [x] b) I want to modify the code, re-build project and see the changes
  - Just something stupid. Print hello wordl from Petr
  - changed `simple.cpp` 
  ``` fprintf(stderr, "Generating token number %d\n", n_decode + 1); ```
  Runs fine.

- [x] c) Next, I want specifically interject into places where RNGs are generated.
  - During inference, sampling
  - Specifically, save each rng generated number to a file

- [x] d) then I want to replace all the custom non-trivial rng generation 
  - (e.g. "sample this custom distribution") with my own implementations using the basic uniform (0,1) rng generator 

- [ ] e) then I want to replace the default (0,1) uniform distribution with my custom provider coming from external api

- [ ] f) Idea: measure the bias in the source distribution based 
  - specifically: see each generated number and see how it changes 

- [ ] g) try / support temperature > 1
  - try min_p lower (0?)
  - try other models - bigger, better