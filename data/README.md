# Data

当前攻击项目只使用两个攻击数据集：

| Folder | Source | Role | Rows |
|---|---|---|---:|
| `advbench/` | AdvBench `harmful_behaviors`（本地 `advbench_harmful_behaviors.csv`） | attack, TTS voice | 520 |
| `jbb/` | JailbreakBench `jbb_full.csv` | attack, TTS voice | 100 |

`data/manifests/` 中的当前 manifest：

```text
advbench.jsonl              # 520 条，full set
advbench_100_pilot.jsonl    # 从 520 条按 stratum 分层抽样的 100 条
jbb_harmful.jsonl           # 100 条 JBB
```

JBB 的主实验使用原输入 safe 子集，保存在：

```text
runs/original_jbb/jbb_safe.jsonl
runs/original_jbb/jbb_safe_lar.jsonl
```

## Build

```bash
# AdvBench：CSV -> 16 kHz mono WAV -> manifest
python scripts/build_advbench_edge_tts_manifest.py

# JailbreakBench：CSV -> 16 kHz mono WAV -> manifest
python scripts/build_jbb_edge_tts_manifest.py
```

## AdvBench

`build_advbench_edge_tts_manifest.py`：

1. 读取 `data/advbench/advbench_harmful_behaviors.csv`（`goal,target`）；
2. 规范化并写出 `data/advbench/advbench_full.csv`（`prompt,stratum,target`）；
3. 用 edge-tts 合成 16 kHz mono WAV 到 `data/advbench/audio/`；
4. 写出 `data/manifests/advbench.jsonl` 和 `advbench.summary.json`。

固定参数：

```text
voice  = en-US-AriaNeural
rate   = +0%
format = 16 kHz, mono, PCM_16
```

`stratum` 使用正则规则分配，共 10 类：

```text
violence, illicit_cyber, harmful_speech, hate, fraud,
deception_rule_evasion, illegal_acts_goods, self_harm,
sexual_exploit, misinformation
```

未命中的记为 `other`。

manifest 字段：

```text
id, local_audio, prompt, source, stratum, target,
tts_engine, tts_voice, tts_rate, audio_sha256, duration_seconds
```

AdvBench 实验**不做原输入过滤**，520 条全部进入攻击，ASR 分母固定为 520。`scripts/run_advbench_suite.sh` 和 `scripts/run_advbench_early_suite.sh` 直接从 `data/manifests/advbench.jsonl` 运行攻击、Llama-Guard 判定和汇总。

## JailbreakBench

`build_jbb_edge_tts_manifest.py`：

1. 读取 `data/jbb/jbb_full.csv`（`prompt,stratum`）；
2. 用 edge-tts 合成 16 kHz mono WAV 到 `data/jbb/audio/`；
3. 写出 `data/manifests/jbb_harmful.jsonl` 和 `jbb_harmful.summary.json`。

TTS 参数与 AdvBench 相同。100 条中，原始 Qwen2-Audio 回答被 Llama-Guard 判为 safe 94 条、unsafe 6 条。主实验使用 94 条 safe 子集，因此：

```text
original input ASR = 0%
reported ASR = 攻击后相对原始输入的增量 ASR
```

## Notes

- 所有 manifest 的 `prompt` 必须是完整的原始有害请求文本；Llama-Guard 使用 `prompt + response` 判定。
- AdvBench 与 JBB 都使用合成语音，论文中需要注明 TTS engine、voice 和 rate。
