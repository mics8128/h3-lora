# Third-party notices

The rectangular Morton decoder and the dynamic symmetric int8 quantization /
Metal 4 TensorOps scheduling design in `h3_shaders.metal` are adapted from
ccv's Metal FlashAttention `NAMatMulKernel` and `NAInt8MatMulKernel`,
distributed under the following BSD-3-Clause license:

Copyright (c) 2010, Liu Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the authors nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## h3.c optimization references (MIT)

The on-demand direct Metal pipeline cache is adapted from jerryfane's
[PR75](https://github.com/antirez/h3.c/pull/75), pinned at
`30b63b51404cae1f2dcc1841abea96f1893c5b49`.
The optional BF16 video VAE projections, attention and typed bias-widening
operation are adapted from justwjx's
[PR73](https://github.com/antirez/h3.c/pull/73), pinned at
`aac7b5ab1b920ede22c7ce55e6139d11bde06173`.
Both source trees carry the MIT license, Copyright (c) 2026 Salvatore
Sanfilippo; the complete permission and warranty notice is retained in
[LICENSE](LICENSE). Model-weight licenses are separate; these code references
do not grant rights to H3 or LightVAE weights.

## ER-SDE numerical algorithm and interoperability references

The ER-SDE mathematical solver family is described in Qinpeng Cui et al.,
[Elucidating the solution space of extended reverse-time SDE for diffusion models](https://arxiv.org/abs/2309.06169).
The [original solver implementation](https://github.com/QinpengCui/ER-SDE-Solver/tree/25ca6c2e6b065754a55694dc6941a3259fa0f082)
is MIT-licensed; its notice is retained below for the numerical solver reference/adaptation.
This fork implements native joint-audiovisual carried state, masks, RNG and checkpoint storage;
it does not bundle the original Python runtime.

MIT License

Copyright (c) 2023 Qinpeng Cui

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Additional numerical compatibility references are separately licensed:

- [ComfyUI `fe298e4e20f1ff26d30e8639267bbd0b57811390`](https://github.com/Comfy-Org/ComfyUI/tree/fe298e4e20f1ff26d30e8639267bbd0b57811390):
  ER-SDE CONST ordering/quadrature, beta-grid behavior and H3 curve/final-layer dtype contracts.
  Its [license is GPLv3](https://github.com/Comfy-Org/ComfyUI/blob/fe298e4e20f1ff26d30e8639267bbd0b57811390/LICENSE), not this repository's MIT license.
- [RES4LYF `56e44755d866c87ff9cb4b6608c8e42155db6ecc`, `sigmas.py`1420–1421](https://github.com/ClownsharkBatwing/RES4LYF/blob/56e44755d866c87ff9cb4b6608c8e42155db6ecc/sigmas.py#L1420-L1421):
  the named `beta57` binding to Beta(.5,.7). Its [license](https://github.com/ClownsharkBatwing/RES4LYF/blob/56e44755d866c87ff9cb4b6608c8e42155db6ecc/LICENSE)
  contains AGPLv3 text plus an additional commercial-service permission restriction;
  do not mislabel it ordinary GPL/MIT or import its runtime under this repository's license.

These are mathematical/behavior compatibility references; their Python runtimes are not
vendored or required for native inference. Pinned source attribution and a review finding
of no wholesale copied source block are not clean-room certification or legal clearance.
Model-weight terms remain separate; see [Eros beta5 source and license notes](EROS_BETA5_SETUP.md#來源與授權).

