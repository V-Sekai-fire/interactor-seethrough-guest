# interactor-seethrough-guest

See-Through (anime illustration -> inpainted semantic layers + depth) as a
godot-sandbox guest ELF whose ggml graph runs on ggml-rd.

## Provenance

Not a fork. Sources in `src/`, `scripts/`, `verify/` and `docs/` are copied from
V-Sekai-archive/interactor-seethrough-ggml at commit
`ae338e268ed8b246f5675b480c3408646774f57e` (Apache-2.0, see LICENSE). Model code is
kept unchanged; `third_party/ggml` is NOT vendored, the guest builds against the
ggml that ggml-rd uses. The archive's measurements (2.12x slower than torch on
ggml-vulkan; f32 activations, no half-precision NORM/GROUP_NORM/GLU) are the
baseline this port must beat.

## Plan

1. Spike: one stage (trans-vae) on ggml-rd, parity vs the archived port on
   `upstream-art/anime_with_caption_cc0_0023.jpg` (cosine > 0.999 plus L2 relative
   error, planted control).
2. Authoritative `supports_op` walk over the built graphs.
3. bf16 GGUFs as reference; Q4_0/Q8_0 mul_mat kernels in ggml-rd after.
4. RFD, guest wiring (read callbacks, pump), speed vs ggml-vulkan.
