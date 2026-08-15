#!/data/data/com.termux/files/usr/bin/bash

set -Eeuo pipefail

PROGRAM="${0##*/}"
REPO="${REPO:-whikao/llama.cpp}"
RELEASE_TAG="${RELEASE_TAG:-llama-android-hexagon-phone-latest}"
ASSET="${ASSET:-llama.cpp-android-hexagon-uma-fit-v4.tar.gz}"
APP_ROOT="${APP_ROOT:-$HOME/.local/share/llama-phone-debug}"
RELEASES_DIR="$APP_ROOT/releases"
CURRENT_LINK="$APP_ROOT/current"
LOG_ROOT="${LOG_ROOT:-/sdcard/htp-debug}"
MODEL="${MODEL:-/sdcard/gguf/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf}"
PROMPT="${PROMPT:-Hello.}"
NDEV="${NDEV:-4}"
DEVICES="${DEVICES:-}"
CTX_SIZE="${CTX_SIZE:-64}"
THREADS="${THREADS:-4}"
TOKENS="${TOKENS:-1}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"
HTP_NGL="${HTP_NGL:-}"
TRACE_START="${TRACE_START:-blk.0.ffn_gate_exps}"
TRACE_COUNT="${TRACE_COUNT:-8}"
DEBUG_K="${DEBUG_K:-192}"
TERMUX_PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
TMP_BASE="${TMPDIR:-$TERMUX_PREFIX/tmp}"
TEMP_DIR=""
LAST_RUN_RC=0

say() {
    printf '[phone-debug] %s\n' "$*"
}

warn() {
    printf '[phone-debug] WARNING: %s\n' "$*" >&2
}

die() {
    printf '[phone-debug] ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage:
  $PROGRAM all [llama-cli arguments...]
  $PROGRAM install
  $PROGRAM run [both|cpu|htp] [llama-cli arguments...]
  $PROGRAM status

The default command is "all". Configuration is supplied with environment
variables. Common variables are MODEL, PROMPT, NDEV, DEVICES, CTX_SIZE,
THREADS, TOKENS, TIMEOUT_SECS, HTP_NGL, LOG_ROOT, and APP_ROOT.
EOF
}

default_devices() {
    local count="$1"
    local result=""
    local i

    for ((i = 0; i < count; i++)); do
        if [[ -n "$result" ]]; then
            result+=","
        fi
        result+="HTP$i"
    done
    printf '%s' "$result"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Missing command: $1. Run: pkg install curl tar coreutils"
}

require_integer() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer: $value"
}

validate_config() {
    [[ "$REPO" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "REPO must use owner/name form: $REPO"
    [[ -n "$RELEASE_TAG" && "$RELEASE_TAG" != */* ]] || die "RELEASE_TAG must be one path component"
    [[ -n "$ASSET" && "$ASSET" != */* ]] || die "ASSET must be a file name"
    require_integer NDEV "$NDEV"
    require_integer CTX_SIZE "$CTX_SIZE"
    require_integer THREADS "$THREADS"
    require_integer TOKENS "$TOKENS"
    require_integer TIMEOUT_SECS "$TIMEOUT_SECS"
    require_integer TRACE_COUNT "$TRACE_COUNT"
    require_integer DEBUG_K "$DEBUG_K"
    ((NDEV >= 1 && NDEV <= 4)) || die "NDEV must be between 1 and 4"
    ((CTX_SIZE >= 1)) || die "CTX_SIZE must be at least 1"
    ((THREADS >= 1)) || die "THREADS must be at least 1"
    [[ -z "$HTP_NGL" || "$HTP_NGL" =~ ^[0-9]+$ ]] || die "HTP_NGL must be empty or a non-negative integer"
    if [[ -z "$DEVICES" ]]; then
        DEVICES="$(default_devices "$NDEV")"
    fi
}

cleanup() {
    local safe_prefix="$TMP_BASE/llama-phone-debug."

    if [[ -n "$TEMP_DIR" && "$TEMP_DIR" == "$safe_prefix"* && -d "$TEMP_DIR" ]]; then
        rm -rf -- "$TEMP_DIR"
    fi
}

trap cleanup EXIT
trap 'exit 130' HUP INT TERM

ensure_storage() {
    if [[ "$MODEL" == /sdcard/* || "$LOG_ROOT" == /sdcard/* ]]; then
        [[ -d /sdcard ]] || die "/sdcard is unavailable. Run termux-setup-storage and accept the Android prompt"
    fi

    mkdir -p "$LOG_ROOT" 2>/dev/null || die "Cannot write $LOG_ROOT. Run termux-setup-storage and accept the Android prompt"
}

validate_archive() {
    tar -tzf "$1" | awk '
        {
            count++
            if ($0 != "llama.cpp" && index($0, "llama.cpp/") != 1) {
                bad = 1
            }
            if ($0 ~ /(^|\/)\.\.(\/|$)/) {
                bad = 1
            }
        }
        END {
            if (count == 0 || bad) {
                exit 1
            }
        }
    '
}

install_release() {
    local base_url="https://github.com/$REPO/releases/download/$RELEASE_TAG"
    local archive
    local checksum_file
    local build_info
    local expected=""
    local actual=""
    local verified=0
    local attempt
    local version
    local staged
    local target
    local current_tmp
    local required

    require_command curl
    require_command tar
    require_command sha256sum
    require_command awk
    require_command mktemp

    mkdir -p "$RELEASES_DIR" "$TMP_BASE"
    TEMP_DIR="$(mktemp -d "$TMP_BASE/llama-phone-debug.XXXXXXXX")"
    archive="$TEMP_DIR/$ASSET"
    checksum_file="$TEMP_DIR/SHA256SUMS"
    build_info="$TEMP_DIR/BUILD_INFO.txt"

    for attempt in 1 2 3; do
        say "Downloading $ASSET (attempt $attempt/3)"
        if ! curl --fail --location --silent --show-error --retry 3 --connect-timeout 30 "$base_url/$ASSET" -o "$archive"; then
            warn "Package download failed"
            continue
        fi
        if ! curl --fail --location --silent --show-error --retry 3 --connect-timeout 30 "$base_url/SHA256SUMS" -o "$checksum_file"; then
            warn "Checksum download failed"
            continue
        fi

        expected="$(awk -v name="$ASSET" '
            {
                file = $2
                sub(/^\*/, "", file)
                if (file == name) {
                    print $1
                    exit
                }
            }
        ' "$checksum_file")"
        actual="$(sha256sum "$archive" | awk '{print $1}')"

        if [[ "$expected" =~ ^[0-9a-fA-F]{64}$ && "${actual,,}" == "${expected,,}" ]]; then
            verified=1
            break
        fi

        warn "Checksum mismatch. The rolling release may be updating; retrying"
        sleep 2
    done

    ((verified == 1)) || die "Unable to download a verified package from $base_url"
    validate_archive "$archive" || die "The package contains an unexpected path"

    if ! curl --fail --location --silent --show-error --retry 2 --connect-timeout 30 "$base_url/BUILD_INFO.txt" -o "$build_info"; then
        warn "BUILD_INFO.txt was not available"
    fi

    version="${actual:0:16}"
    staged="$TEMP_DIR/unpacked"
    target="$RELEASES_DIR/$version"
    mkdir -p "$staged"
    tar -xzf "$archive" -C "$staged"

    for required in \
        bin/llama-cli \
        lib/libggml-cpu.so \
        lib/libggml-hexagon.so \
        lib/libggml-htp-v81.so; do
        [[ -f "$staged/llama.cpp/$required" ]] || die "Package is missing $required"
    done
    chmod +x "$staged/llama.cpp/bin/llama-cli"
    cp "$checksum_file" "$staged/llama.cpp/PHONE_SHA256SUMS"
    if [[ -s "$build_info" ]]; then
        cp "$build_info" "$staged/llama.cpp/PHONE_BUILD_INFO.txt"
    fi

    if [[ ! -d "$target" ]]; then
        mv "$staged/llama.cpp" "$target"
    fi
    [[ -x "$target/bin/llama-cli" ]] || die "Installed release is incomplete: $target"

    if [[ -e "$CURRENT_LINK" && ! -L "$CURRENT_LINK" ]]; then
        die "$CURRENT_LINK exists and is not a symbolic link"
    fi
    current_tmp="$APP_ROOT/.current.$BASHPID"
    ln -s "releases/$version" "$current_tmp"
    mv -Tf "$current_tmp" "$CURRENT_LINK"

    cleanup
    TEMP_DIR=""
    say "Installed release $version at $target"
}

current_release() {
    local resolved

    [[ -L "$CURRENT_LINK" ]] || die "No installed release. Run: $PROGRAM install"
    resolved="$(readlink -f "$CURRENT_LINK")"
    [[ "$resolved" == "$RELEASES_DIR/"* ]] || die "Current release points outside $RELEASES_DIR"
    [[ -x "$resolved/bin/llama-cli" ]] || die "llama-cli is missing from $resolved"
    printf '%s' "$resolved"
}

write_metadata() {
    local output="$1"
    local release="$2"

    {
        printf 'captured_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'release=%s\n' "${release##*/}"
        printf 'model=%s\n' "$MODEL"
        printf 'model_size_bytes=%s\n' "$(stat -c %s "$MODEL" 2>/dev/null || printf unknown)"
        printf 'devices=%s\n' "$DEVICES"
        printf 'ndev=%s\n' "$NDEV"
        printf 'ctx_size=%s\n' "$CTX_SIZE"
        printf 'threads=%s\n' "$THREADS"
        printf 'tokens=%s\n' "$TOKENS"
        printf 'uname=%s\n' "$(uname -a)"
        printf 'llama_cli_sha256=%s\n' "$(sha256sum "$release/bin/llama-cli" | awk '{print $1}')"
        if command -v getprop >/dev/null 2>&1; then
            printf 'product=%s\n' "$(getprop ro.product.manufacturer) $(getprop ro.product.model)"
            printf 'soc=%s\n' "$(getprop ro.soc.manufacturer) $(getprop ro.soc.model)"
            printf 'android=%s\n' "$(getprop ro.build.version.release)"
            printf 'fingerprint=%s\n' "$(getprop ro.build.fingerprint)"
        fi
        if [[ -f "$release/PHONE_BUILD_INFO.txt" ]]; then
            printf '\n[build_info]\n'
            cat "$release/PHONE_BUILD_INFO.txt"
        fi
        printf '\n[hexagon_environment]\n'
        printf 'GGML_HEXAGON_ARCH=%s\n' "$GGML_HEXAGON_ARCH"
        printf 'GGML_HEXAGON_NDEV=%s\n' "$GGML_HEXAGON_NDEV"
        printf 'GGML_HEXAGON_MMID_RAW_Q4_0=%s\n' "$GGML_HEXAGON_MMID_RAW_Q4_0"
        printf 'GGML_HEXAGON_HOSTBUF=%s\n' "$GGML_HEXAGON_HOSTBUF"
        printf 'GGML_HEXAGON_VERBOSE=%s\n' "$GGML_HEXAGON_VERBOSE"
        printf 'GGML_HEXAGON_DEBUG_K=%s\n' "$GGML_HEXAGON_DEBUG_K"
        printf 'GGML_HEXAGON_TRACE_START=%s\n' "$GGML_HEXAGON_TRACE_START"
        printf 'GGML_HEXAGON_TRACE_COUNT=%s\n' "$GGML_HEXAGON_TRACE_COUNT"
    } > "$output"
}

run_logged() {
    local label="$1"
    local log_path="$2"
    local rc
    shift 2

    {
        printf '=== %s ===\n' "$label"
        printf 'Started: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'Command:'
        printf ' %q' "$@"
        printf '\n\n'
    } | tee "$log_path"

    set +e
    if ((TIMEOUT_SECS > 0)); then
        timeout --signal=INT --kill-after=30s "${TIMEOUT_SECS}s" "$@" 2>&1 | tee -a "$log_path"
        rc=${PIPESTATUS[0]}
    else
        "$@" 2>&1 | tee -a "$log_path"
        rc=${PIPESTATUS[0]}
    fi
    set -e

    printf '\nExit code: %s\n' "$rc" | tee -a "$log_path"
    printf '%s\n' "$rc" > "$log_path.exit-code"
    LAST_RUN_RC=$rc
}

analyze_htp_log() {
    local log_path="$1"
    local output_dir="$2"

    if [[ ! -f "$log_path" ]]; then
        return
    fi

    grep -aE 'DBG_V[0-9]+_|MUL_MAT_ID' "$log_path" > "$output_dir/debug-trace.log" || true
    tail -n 200 "$log_path" > "$output_dir/htp-tail.log"

    awk '
        function value(key,    i, text) {
            for (i = 1; i <= NF; i++) {
                if (index($i, key "=") == 1) {
                    text = substr($i, length(key) + 2)
                    return text
                }
            }
            return ""
        }
        function integer_value(key,    text) {
            text = value(key)
            gsub(/[^0-9-]/, "", text)
            if (text == "") {
                return -1
            }
            return text + 0
        }
        function ne2_value(    text, dims) {
            text = value("src_ne")
            gsub(/[\[\](){}]/, "", text)
            split(text, dims, ",")
            gsub(/[^0-9-]/, "", dims[3])
            if (dims[3] == "") {
                return -1
            }
            return dims[3] + 0
        }
        function emit(ne2, label) {
            if (total[ne2] > 0) {
                printf "%s\t%d\t%d\t%d\t%d\n", label, total[ne2], bad_src[ne2] + 0, bad_dst[ne2] + 0, finite_to_bad[ne2] + 0
            }
        }
        /DBG_V120_MMID_SLICE/ {
            gsub(/[[:space:]]*=[[:space:]]*/, "=", $0)
            ne2 = ne2_value()
            src_bad = integer_value("src_nonfinite")
            dst_bad = integer_value("dst_nonfinite")
            total[ne2]++
            if (src_bad > 0) {
                bad_src[ne2]++
            }
            if (dst_bad > 0) {
                bad_dst[ne2]++
            }
            if (src_bad == 0 && dst_bad > 0) {
                finite_to_bad[ne2]++
            }
        }
        END {
            print "ne2\tslices\tsrc_nonfinite_slices\tdst_nonfinite_slices\tfinite_src_to_bad_dst"
            emit(2, "2")
            emit(10, "10")
            emit(-1, "unknown")
            for (ne2 in total) {
                if (ne2 != 2 && ne2 != 10 && ne2 != -1) {
                    emit(ne2, ne2)
                }
            }
        }
    ' "$log_path" > "$output_dir/mmid-summary.tsv"
}

make_summary() {
    local output_dir="$1"
    local mode="$2"
    local cpu_rc="$3"
    local htp_rc="$4"
    local summary="$output_dir/summary.md"
    local marker_count=0

    if [[ -f "$output_dir/debug-trace.log" ]]; then
        marker_count="$(wc -l < "$output_dir/debug-trace.log")"
    fi

    {
        printf '# Termux Hexagon debug summary\n\n'
        printf -- '- Mode: %s\n' "$mode"
        printf -- '- CPU exit code: %s\n' "$cpu_rc"
        printf -- '- HTP exit code: %s\n' "$htp_rc"
        printf -- '- Extracted debug lines: %s\n\n' "$marker_count"
        printf '## MUL_MAT_ID slices\n\n'
        printf '| ne2 | slices | src nonfinite | dst nonfinite | finite src to bad dst |\n'
        printf '| ---: | ---: | ---: | ---: | ---: |\n'
        if [[ -f "$output_dir/mmid-summary.tsv" ]]; then
            awk -F '\t' 'NR > 1 { printf "| %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5 }' "$output_dir/mmid-summary.tsv"
        fi
        printf '\n## Metadata\n\n```text\n'
        cat "$output_dir/metadata.txt"
        printf '```\n\n'
        printf 'Attach share-this.tar.gz to the chat. Keep htp.log on the phone in case the full log is needed.\n'
    } > "$summary"
}

make_share_bundle() {
    local output_dir="$1"
    local files=(summary.md metadata.txt)
    local candidate

    for candidate in mmid-summary.tsv debug-trace.log htp-tail.log cpu-tail.log cpu.log.exit-code htp.log.exit-code; do
        if [[ -f "$output_dir/$candidate" ]]; then
            files+=("$candidate")
        fi
    done

    (
        cd "$output_dir"
        tar -czf share-this.tar.gz "${files[@]}"
    )
}

run_suite() {
    local mode="$1"
    local release
    local run_id
    local output_dir
    local cpu_rc="not-run"
    local htp_rc="not-run"
    local cli
    local -a common_args
    local -a htp_args
    shift

    case "$mode" in
        both|cpu|htp) ;;
        *) die "Run mode must be both, cpu, or htp: $mode" ;;
    esac

    require_command readlink
    require_command sha256sum
    require_command awk
    require_command grep
    require_command tail
    require_command tee
    require_command tar
    if ((TIMEOUT_SECS > 0)); then
        require_command timeout
    fi

    ensure_storage
    [[ -f "$MODEL" ]] || die "Model not found: $MODEL"
    release="$(current_release)"
    cli="$release/bin/llama-cli"
    run_id="$(date -u +%Y%m%d-%H%M%S)-${release##*/}"
    output_dir="$LOG_ROOT/$run_id"
    mkdir -p "$output_dir"

    export LD_LIBRARY_PATH="/vendor/lib64:/vendor/lib:$release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ADSP_LIBRARY_PATH="$release/lib${ADSP_LIBRARY_PATH:+:$ADSP_LIBRARY_PATH}"
    export GGML_HEXAGON_ARCH="${GGML_HEXAGON_ARCH:-81}"
    export GGML_HEXAGON_NDEV="$NDEV"
    export GGML_HEXAGON_MMID_RAW_Q4_0="${GGML_HEXAGON_MMID_RAW_Q4_0:-1}"
    export GGML_HEXAGON_HOSTBUF="${GGML_HEXAGON_HOSTBUF:-1}"
    export GGML_HEXAGON_VERBOSE="${GGML_HEXAGON_VERBOSE:-1}"
    export GGML_HEXAGON_DEBUG_K="$DEBUG_K"
    export GGML_HEXAGON_TRACE_START="$TRACE_START"
    export GGML_HEXAGON_TRACE_COUNT="$TRACE_COUNT"

    write_metadata "$output_dir/metadata.txt" "$release"
    common_args=(-c "$CTX_SIZE" -t "$THREADS" -n "$TOKENS" -v -m "$MODEL" -p "$PROMPT")

    if [[ "$mode" == both || "$mode" == cpu ]]; then
        say "Running CPU baseline"
        run_logged "CPU baseline" "$output_dir/cpu.log" "$cli" --device none -ngl 0 "${common_args[@]}" "$@"
        cpu_rc="$LAST_RUN_RC"
        tail -n 200 "$output_dir/cpu.log" > "$output_dir/cpu-tail.log"
    fi

    if [[ "$mode" == both || "$mode" == htp ]]; then
        say "Running Hexagon HTP on $DEVICES"
        htp_args=("$cli" --device "$DEVICES" "${common_args[@]}")
        if [[ -n "$HTP_NGL" ]]; then
            htp_args+=(-ngl "$HTP_NGL")
        fi
        htp_args+=("$@")
        run_logged "Hexagon HTP" "$output_dir/htp.log" "${htp_args[@]}"
        htp_rc="$LAST_RUN_RC"
        analyze_htp_log "$output_dir/htp.log" "$output_dir"
    fi

    make_summary "$output_dir" "$mode" "$cpu_rc" "$htp_rc"
    make_share_bundle "$output_dir"
    say "Logs: $output_dir"
    say "Attach this file to the chat: $output_dir/share-this.tar.gz"

    if [[ "$cpu_rc" != not-run && "$cpu_rc" != 0 ]]; then
        return "$cpu_rc"
    fi
    if [[ "$htp_rc" != not-run && "$htp_rc" != 0 ]]; then
        return "$htp_rc"
    fi
}

show_status() {
    local release

    release="$(current_release)"
    printf 'Current release: %s\n' "$release"
    printf 'llama-cli: %s\n' "$release/bin/llama-cli"
    printf 'Model: %s\n' "$MODEL"
    printf 'Log root: %s\n' "$LOG_ROOT"
    if [[ -f "$release/PHONE_BUILD_INFO.txt" ]]; then
        printf '\n'
        cat "$release/PHONE_BUILD_INFO.txt"
    fi
}

main() {
    local command_name="${1:-all}"
    local mode

    validate_config

    case "$command_name" in
        all)
            shift || true
            install_release
            run_suite both "$@"
            ;;
        install)
            (($# == 1)) || die "install does not accept arguments"
            install_release
            ;;
        run)
            shift
            mode="${1:-both}"
            if (($# > 0)); then
                shift
            fi
            run_suite "$mode" "$@"
            ;;
        status)
            (($# == 1)) || die "status does not accept arguments"
            show_status
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            usage >&2
            die "Unknown command: $command_name"
            ;;
    esac
}

main "$@"
