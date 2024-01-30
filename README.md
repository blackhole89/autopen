# Autopen

(As in [the device](https://en.wikipedia.org/wiki/Autopen), and a [pen](https://en.wikipedia.org/wiki/Pen_%28enclosure%29) for [assorted](https://en.wikipedia.org/wiki/LLaMA) [macrofauna](https://lmsys.org/blog/2023-03-30-vicuna/))

[![Demonstration video](https://img.youtube.com/vi/1O1T2q2t7i4/maxresdefault.jpg)](https://www.youtube.com/watch?v=1O1T2q2t7i4)

This is not release-quality software. Expect crashes, unexpected behaviour and terrible code.

Everything is licensed under the GNU GPL version 3. At this stage, I don't intend to accept pull requests.

### To build
```
git clone https://github.com/ggerganov/llama.cpp.git
(cd llama.cpp && patch -p1 ../llama.patch && cmake . && make llama)
cmake .
make
```
Dependencies may include gtkmm-3.0 and more.
### To run
```
./cmake-build-Release/output/autopen
```
The model location is hardcoded to `./llama.cpp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (so in particular, this file is assumed to be present), but feel free to change this.

