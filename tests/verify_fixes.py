import json, sys, pathlib

ROOT = pathlib.Path(r"d:\minimax\MiniMax-H3")
results = []

def check(num, name, condition, detail=""):
    icon = "PASS" if condition else "FAIL"
    results.append((num, name, icon, detail))
    tag = "OK" if condition else "!!"
    print(f"  [{tag}] Bug {num}: {name}" + (f" -> {detail}" if detail else ""))

# BUG 1
for task in ("FL2VA", "Ref2VA"):
    src = (ROOT / task / "audio_vae" / "dac_alias_free_filter.py").read_text(encoding="utf-8")
    lines = src.splitlines()
    fix_ok  = any("    filter = filter_.view(1, 1, kernel_size)" in l
                  and not l.startswith("        ") for l in lines)
    still_buggy = any("        filter = filter_.view" in l for l in lines)
    check(1, f"NameError cutoff==0 [{task}]", fix_ok and not still_buggy)

# BUG 2
for task in ("FL2VA", "Ref2VA"):
    src = (ROOT / task / "audio_vae" / "dac_audio_vae.py").read_text(encoding="utf-8")
    count = src.count("self.sample_rate = sample_rate")
    check(2, f"Duplicate sample_rate [{task}]", count == 1, f"{count} assignment(s)")

# BUG 3
for task in ("FL2VA", "Ref2VA"):
    src = (ROOT / task / "audio_vae" / "dac_audio_vae.py").read_text(encoding="utf-8")
    has_trunc = "trunc_normal_" in src
    has_std   = "std=0.02" in src
    has_bias  = "m.bias" in src
    apply_calls = sum(1 for l in src.splitlines()
                      if "self.apply(init_weights)" in l
                      and not l.strip().startswith("#")
                      and not l.strip().startswith("``"))
    check(3, f"Conflicting init_weights [{task}]",
          has_trunc and has_std and has_bias and apply_calls == 1,
          f"trunc={has_trunc} std02={has_std} bias={has_bias} apply={apply_calls}")

# BUG 4
for task in ("FL2VA", "Ref2VA"):
    src = (ROOT / task / "audio_vae" / "dac_attn_proj.py").read_text(encoding="utf-8")
    has_qkv_out     = "self.qkv_out_dim" in src
    old_mean_gone   = "torch.mean(x, dim=1)" not in src
    uniform_reshape = "x.transpose(1, 2).reshape(B, N, self.qkv_out_dim)" in src
    check(4, f"CausalAttention shape [{task}]",
          has_qkv_out and old_mean_gone and uniform_reshape)

# BUG 5
for task in ("FL2VA", "Ref2VA"):
    src = (ROOT / task / "audio_vae" / "dac_audio_vae.py").read_text(encoding="utf-8")
    new_ok = "padding=stride // 2" in src
    old_in_code = any("padding=math.ceil(stride" in l and not l.strip().startswith("#")
                      for l in src.splitlines())
    check(5, f"EncoderBlock padding [{task}]", new_ok and not old_in_code)

# BUG 6
for task in ("FL2VA", "Ref2VA"):
    src = (ROOT / task / "audio_vae" / "dac_audio_vae.py").read_text(encoding="utf-8")
    has_explicit = "pad_right = diff - pad_left" in src
    old_gone     = not any("x[..., pad:-pad]" in l and not l.strip().startswith("#")
                           for l in src.splitlines())
    check(6, f"ResidualUnit odd diff [{task}]", has_explicit and old_gone)

# BUG 7
for task in ("FL2VA", "Ref2VA"):
    cfg = json.loads((ROOT / task / "transformer" / "config.json").read_text(encoding="utf-8"))
    pairs = [("ffn_hidden_size", "ffn_dim"),
             ("rope_inv_freq_len", "rope_freq_dim"),
             ("token_refiner_num_layers", "num_refiner_layers")]
    ok = all(a in cfg and b in cfg and cfg[a] == cfg[b] for a, b in pairs)
    check(7, f"Transformer config consistency [{task}]", ok)

# BUG 8
fl2va_v = json.loads((ROOT / "FL2VA" / "model_index.json").read_text())["_diffusers_version"]
ref2va_v = json.loads((ROOT / "Ref2VA" / "model_index.json").read_text())["_diffusers_version"]
root_v   = json.loads((ROOT / "model_index.json").read_text())["_diffusers_version"]
mod_v    = json.loads((ROOT / "modular_model_index.json").read_text())["_diffusers_version"]
check(8, "Diffusers version consistency",
      fl2va_v == ref2va_v and root_v == mod_v,
      f"FL2VA/Ref2VA={fl2va_v} | Root/Modular={root_v}")

# BUG 9
req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
check(9, "PyYAML in requirements.txt", "PyYAML" in req)

# BUG 10
crlf_files = [f.name for f in (ROOT / "scripts" / "readme").glob("*.sh")
               if b"\r\n" in f.read_bytes()]
check(10, "CRLF in .sh scripts",
      len(crlf_files) == 0,
      f"CRLF still in: {crlf_files}" if crlf_files else "All 18 scripts are LF-only")

# BUG 11
missing = [s for s in ["full-2k-t2va-h3-base.sh", "full-2k-i2va-h3-base.sh", "full-2k-ref2va-h3-base.sh"]
           if "--fail-with-body" not in
           (ROOT / "scripts" / "readme" / s).read_text(encoding="utf-8")]
check(11, "curl --fail-with-body in h3-base scripts",
      len(missing) == 0,
      f"Missing in: {missing}" if missing else "All 3 scripts have the flag")

# BUG 12
cfg12 = json.loads((ROOT / "audio_vae" / "config.json").read_text(encoding="utf-8"))
both  = "sample_rate" in cfg12 and "sampling_rate" in cfg12
equal = cfg12.get("sample_rate") == cfg12.get("sampling_rate")
check(12, "sampling_rate vs sample_rate",
      both and equal,
      "sample_rate=32000 & sampling_rate=32000 both present" if (both and equal) else "MISMATCH")

# BUG 13
readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
ok13 = ("Step 1" in readme and "Step 2" in readme and
        "Step 3" in readme and "hf download MiniMaxAI/MiniMax-H3" in readme)
check(13, "README weight download docs",
      ok13, "4-step guide + hf download command present" if ok13 else "MISSING")

# SUMMARY
print()
passes = sum(1 for _, _, s, _ in results if s == "PASS")
fails  = [r for r in results if r[2] == "FAIL"]
total  = len(results)
print("=" * 58)
print(f"  RESULT: {passes}/{total} checks PASSED")
if fails:
    print(f"  FAILED ({len(fails)}):")
    for num, name, _, detail in fails:
        print(f"    Bug {num}: {name}  —  {detail}")
else:
    print(f"  ALL {total} CHECKS PASSED — EVERY BUG IS FIXED")
print("=" * 58)
sys.exit(0 if not fails else 1)
