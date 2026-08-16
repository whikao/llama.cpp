# Android Hexagon debug from Termux

This directory supports a phone-only loop:

1. GitHub Actions builds the Android CPU and Hexagon package.
2. The workflow updates a rolling prerelease with a stable download URL.
3. `phone-debug.sh` downloads and verifies the package, installs it under the Termux home directory, runs CPU and HTP checks, and writes a small log bundle to shared storage.

No computer, ADB, root access, or local Android build toolchain is required.

## One-time Termux setup

Use a current Termux build, then run:

```sh
pkg update
pkg install curl tar coreutils
termux-setup-storage
```

Accept the Android storage permission prompt. Put the model at:

```text
/sdcard/gguf/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf
```

Install the helper after this change is on the repository branch you use:

```sh
curl -fL https://raw.githubusercontent.com/whikao/llama.cpp/master/scripts/snapdragon/termux/phone-debug.sh -o "$HOME/phone-debug.sh"
chmod 700 "$HOME/phone-debug.sh"
```

## Build on GitHub

Open the repository in a browser and select:

```text
Actions -> Build llama.cpp Android CPU + Hexagon (UMA fit v4) -> Run workflow
```

Wait for the run to finish. The last step publishes these files under the rolling prerelease tag `llama-android-hexagon-phone-latest`:

- `llama.cpp-android-hexagon-uma-fit-v4.tar.gz`
- `SHA256SUMS`
- `BUILD_INFO.txt`

The Termux helper retries if it catches the rolling release while the files are being updated.

## Install and run

Run the CPU baseline and the four-device HTP test:

```sh
MODEL=/sdcard/gguf/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf \
PROMPT='Hello.' \
"$HOME/phone-debug.sh" all
```

The binary is installed in a versioned directory below `$HOME/.local/share/llama-phone-debug`. It is not installed on `/sdcard`, because shared storage is commonly mounted with execution restrictions.

After the first install, run only HTP with:

```sh
"$HOME/phone-debug.sh" run htp
```

Useful commands:

```sh
"$HOME/phone-debug.sh" install
"$HOME/phone-debug.sh" run cpu
"$HOME/phone-debug.sh" run both
"$HOME/phone-debug.sh" status
```

Extra arguments after the run mode are passed to `llama-cli`:

```sh
"$HOME/phone-debug.sh" run htp --no-mmap --poll 1000
```

Set `HTP_NGL=99` if the tested branch requires an explicit layer offload count:

```sh
HTP_NGL=99 "$HOME/phone-debug.sh" run htp
```

## Configuration

The main environment variables are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL` | `/sdcard/gguf/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf` | GGUF model path |
| `PROMPT` | `Hello.` | Test prompt |
| `NDEV` | `4` | Number of logical HTP devices |
| `DEVICES` | Derived from `NDEV` | Comma-separated llama device list |
| `CTX_SIZE` | `64` | Context size |
| `THREADS` | `4` | CPU thread count |
| `TOKENS` | `1` | Tokens to generate |
| `TIMEOUT_SECS` | `0` | Per-run timeout; disabled by default to match direct Termux execution |
| `HTP_NGL` | Empty | Optional explicit HTP layer count |
| `NHMX` | `1` | Enable HMX; use `0` for an HVX-only diagnostic run |
| `TRACE_START` | `blk.0.ffn_gate_exps` | Hexagon trace start tensor |
| `TRACE_COUNT` | `8` | Trace checkpoint count |
| `DEBUG_K` | `192` | Hexagon debug K value |
| `MMID_DEBUG` | `0` | Enable the MMID runtime diagnostic controls |
| `MMID_QUANT_MODE` | `auto` | Activation quantizer: `auto`, `block`, or `row` |
| `LOG_ROOT` | `/sdcard/htp-debug` | Shared log directory |
| `APP_ROOT` | `$HOME/.local/share/llama-phone-debug` | Versioned installation root |

The script sets the current debug environment automatically:

```text
GGML_HEXAGON_ARCH=81
GGML_HEXAGON_NDEV=4
GGML_HEXAGON_NHMX=1
GGML_HEXAGON_MMID_RAW_Q4_0=1
GGML_HEXAGON_HOSTBUF=1
GGML_HEXAGON_VERBOSE=1
GGML_HEXAGON_MMID_DEBUG=0
GGML_HEXAGON_MMID_QUANT_MODE=auto
GGML_HEXAGON_DEBUG_K=192
GGML_HEXAGON_TRACE_START=blk.0.ffn_gate_exps
GGML_HEXAGON_TRACE_COUNT=8
```

The v10.25 low-memory HMX path keeps `GGML_HEXAGON_MMID_RAW_Q4_0=1`. It
converts only the current HMX weight chunk from raw GGUF Q4_0 into tiled form
inside the two existing VTCM work areas. The independent 32-row layout
transforms run on the existing HTP worker pool; the transformed bytes and HMX
math are unchanged from the correct v10.23 result. Do not set the variable to
`0` on a memory-constrained phone: normal HTP_REPACK placement needs roughly
15 GiB for this model and may be killed by Android.

The v10.21 diagnostic controls can switch the MMID activation quantizer without
another GitHub Actions build. For a focused A/B run, use one of:

```bash
MMID_QUANT_MODE=block "$HOME/phone-debug.sh" run htp
MMID_QUANT_MODE=row "$HOME/phone-debug.sh" run htp
```

`auto` keeps the normal selection rule. `block` and `row` only affect the
experimental raw-Q4_0 `MUL_MAT_ID` path while the debug-control packet is
enabled.

The `ne2=10` case uses the HMX `MUL_MAT_ID` implementation, so the HVX
`block`/`row` selector does not affect it. Set `NHMX=0` only when an HVX-only
comparison is requested.

The helper also passes `--single-turn`. Chat models can automatically enable conversation mode; single-turn mode exits after the predefined prompt instead of waiting at the `>` prompt. The live raw log path is printed before each command starts. `TIMEOUT_SECS=0` is the default because the direct Termux command is the known-good launch form; set a positive timeout explicitly only when needed.

Before launching `llama-cli`, the helper changes its working directory to the installed `lib` directory. llama.cpp discovers dynamic backend plug-ins in the executable directory and current working directory; `LD_LIBRARY_PATH` alone resolves shared-library dependencies but does not add a backend discovery directory. This is required because `llama-cli` is installed in `bin` while `libggml-hexagon.so` is installed in `lib`.

## Results

Each run creates a timestamped directory below `/sdcard/htp-debug` containing:

- `cpu.log` and `htp.log`: full command output.
- `debug-trace.log`: extracted `DBG_*` and `MUL_MAT_ID` lines.
- `mmid-summary.tsv`: counts for the `DBG_V120_MMID_SLICE` checkpoints, including finite source slices that produced nonfinite destination slices.
- `DBG_V121_MMID_*` lines in the raw/debug trace: per-slice quantizer mode,
  source/Q8 fingerprints, expert/Q4 offsets and fingerprints, and exact dot outputs.
- `DBG_V125_HMX_RAW_PROFILE` lines: accumulated raw-HMX phase timings for each
  `MUL_MAT_ID`, including raw DMA, layout conversion, dequantization and HMX
  compute.
- `summary.md`: exit codes, the MMID table, build identity, phone identity, and active debug settings.
- `share-this.tar.gz`: the small bundle to attach to the debugging chat.

The checksum verifies that the package and the workflow checksum match. It protects against transfer errors and update races; it does not replace trust in the repository and GitHub Actions configuration.
