# Autopen

A text editor that lets you view the text through the eyes of an LLM, see what it expects and what it finds surprising, generate continuations,
and seamlessly explore different alternatives at every point - as in [the device](https://en.wikipedia.org/wiki/Autopen), and a [pen](https://en.wikipedia.org/wiki/Pen_%28enclosure%29) for [assorted](https://en.wikipedia.org/wiki/LLaMA) [macrofauna](https://lmsys.org/blog/2023-03-30-vicuna/).

## About

[![Demonstration video](https://img.youtube.com/vi/GXWZPpVI0zU/maxresdefault.jpg)](https://www.youtube.com/watch?v=GXWZPpVI0zU)

LLMs are essentially functions that produce, given any piece of text, a probability distribution over words and word fragments that may immediately follow it.
This true nature is hidden by the way in which LLMs are usually deployed: the LLM is applied to an input string to obtain a distribution, the distribution is *sampled* in some way to pick a single word or fragment to attach, and the resulting extended string is fed back as an input, over and over again. Thus, the LLM becomes a device that produces a single possibly endless stream of text, which (depending on the sampling procedure) may be nondeterministic.

The abstraction of LLMs into "writing machines" is certainly compelling, and has by now given rise to endless forms most beautiful of applications. Used in this way,
the model becomes a slightly unruly building block without regard to the probabilities of each choice that is made or the possibilities that the sampler discards. However, to really understand how a given model ticks, how its "thought patterns" emerge and what gives rise to its mistakes and idiosyncrasies, we need a way to look at that data.

Autopen was created for that purpose. It can:

* Edit text.

* Load and execute any LLM in the GGUF format that [llama.cpp](https://github.com/ggerganov/llama.cpp) can run.

* Highlight the text in real time to visualise which tokens are "surprising" to the model (have low probability).

* Generate and simultaneously display multiple continuations at each point in the text, ordered by descending probability according to the LLM.

* Flip through these continuations (Alt-⬆⬇) and emit them into the buffer (Alt-⮕). 

To see this in practice, see the [old demonstration video](https://www.youtube.com/watch?v=1O1T2q2t7i4) (from before the UI was ported to imgui).

This project is powered by [llama.cpp](https://github.com/ggerganov/llama.cpp), [dear imgui](https://github.com/ocornut/imgui) and [imgui-filebrowser](https://github.com/AirGuanZ/imgui-filebrowser/), as well as SDL2 and OpenGL.

The code that is original to this project is licensed under the GNU GPL v3.

### Windows binary build

A Windows binary build can be downloaded [here](https://www.cattheory.net/files/autopen.zip). You may need the NVIDIA CUDA runtime from [this file](https://www.cattheory.net/files/cuda.zip).

If these builds do not work on your system, e.g. because your NVIDIA drivers or GPU are too old or non-existent, you can
also obtain an alternative build of llama.cpp, possibly along with a matching `cudart(...).zip`, from among [the official builds](https://github.com/ggerganov/llama.cpp/releases/tag/b4589). Just download the appropriate binary package and put *all* DLLs in it in the same
folder as `autopen.exe`, potentially replacing the bundled ones.

### Known limitations

This is not release-quality software. Expect crashes, unexpected behaviour and terrible code.

* Currently, models using SPM tokenization (such as Phi-3.5) are broken due to bizarre space-prefixing behaviour. This can be worked around with with a simple patch to llama.cpp, but that is a bit of a deployment nightmare.

* Crashes due to out-of-memory conditions and bugs are not handled gracefully.

* Save/load function for the buffer is a TODO. Copypaste to/from the text editor of your choice.

* Due to imgui's rendering model, there is nontrivial idle CPU usage (which grows with your monitor's framerate). A smart throttling scheme would be useful.

* llama.cpp's batching is surprisingly not quite monoidal (evaluating a batch of n tokens followed by a batch of m tokens gives slightly different results from evaluating a single batch of n+m tokens), which can lead to nondeterministic results. I do not know if this is a bug on our end, llama.cpp's, or a mathematical inevitability.

### To build
Currently a mess.

On Linux (possibly outdated):
```
git clone https://github.com/ocornut/imgui.git
cd imgui
git checkout docking
cd ..
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
# patch -p1 < ../llama.patch
cmake . && make llama && make common
cd ..
cmake .
make
```
Dependencies may include SDL2 and more.

On Windows: try the VS2022 .sln file (and vcpkg). imgui should be cloned as on Linux, and llama.cpp should be cloned and built with a configuration matching the Release/Debug setting you pick (as libcommon has to be statically linked). Some adjustment of include and library directories will probably be necessary.

### To run
Under Linux:
```
./cmake-build-Release/output/autopen
```

Under Windows: run `autopen.exe` in a folder containing the .otf/.ttc fonts and all required DLLs.
