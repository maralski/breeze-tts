# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Writes tokenizer_fixtures.json and prompt_fixtures.json next to this file.

Environment: the repo's .venv (Python 3, transformers 4.57.3, tokenizers 0.22.2,
torch 2.9.1). CHECKPOINT is the Breeze-TTS-2 snapshot directory; REF is the
reference inference code (the directory holding breeze_infer/):

    .venv/bin/python make_text_fixtures.py CHECKPOINT REF

Tokenizer ids come from AutoTokenizer, which is what the reference calls, and the
script asserts that the bare `tokenizers` library agrees. Prompts come from the
reference's own templates.prepare_inputs, with the audio tokenizer replaced by
fixed random codes. The script also checks the claim that encoding each prompt
segment on its own equals the reference's joint re-encode, and fails on a
counterexample.
"""
import json, os, random, sys
from types import SimpleNamespace

CHECKPOINT, REF = (os.path.expanduser(a) for a in sys.argv[1:3])
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REF)
from tokenizers import Tokenizer
from transformers import AutoTokenizer
import torch
from breeze_infer import templates

tok = AutoTokenizer.from_pretrained(CHECKPOINT)
raw = Tokenizer.from_file(os.path.join(CHECKPOINT, "tokenizer.json"))
rng = random.Random(20261117)

specials = ["[S0]", "[S1]", "[S5]", "[S9]", "<ins_bos>", "<ins_eos>", "<|AUDIO|>", "<|audio_eos|>",
            "<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "[multimodal]", "<start_of_turn>", "<end_of_turn>"]
corpus = [
    "", " ", "  ", "   ", "\n", "\n\n", "\r\n", "\t", "\t\t", "a", "A", "Hello world.", "Hello  world", " Hello", "Hello ",
    "  leading and trailing  ", "multiple     spaces   inside", "hello\n\nworld  ", "line one\nline two\r\nline three",
    "tab\tseparated\tvalues", "mixed \t \n whitespace", "a\n \nb", "x \n\n y",
    "The quick brown fox jumps over the lazy dog.",
    "It is good to hear your voice again after all this time.",
    "Welcome aboard. Your journey begins now.",
    "We need to discuss what happened last night.",
    "I'm sure you've heard they'll go, it's John's. DON'T SHOUT!",
    "‘quoted’ “double” \"plain\" 'single' «guillemets» „low“",
    "The S&P 500 fell 2.3% to 5,123.45 on Tuesday; the ASX 200 rose $1.2bn (A$1,234,567.89).",
    "Call 1-800-555-0199 or +61 3 9123 4567 before 9:30am, 24/7.",
    "In 2026, 3.14159 ≈ π and 1/3 ≈ 0.333… ½ ¾ ²³ ① ⅷ",
    "Numbers 0123456789 and ١٢٣ and 一二三 and 12345678901234567890",
    "He said—quite loudly—that it was over. A dash – en dash – and a hyphen-ated word.",
    "Visit https://news.example.com/national/city/story-20261105.html?ref=rss#top now.",
    "Email jane.doe+news@example.co.uk or see www.example.org/path.",
    "Mr. Smith met Dr. Jones vs. Prof. Lee, etc. U.S. officials e.g. today i.e. now.",
    "____---====!!!???...,,,;;;", "!!!!!", "......", "?!?!", "…", "——", "~~~", "***", "###", "@@@", "$$$",
    "C++ C# F# .NET node.js 3D 4K 5G COVID-19 H5N1",
    "x" * 300, "ab" * 200, "1" * 50, "a" * 1000, "的" * 200, "ha" * 150, " " * 40 + "x", "x" + " " * 40,
    "中文测试：今天天气很好。我们去公园吧！", "这是中文参考音频的准确文字稿。", "没想到过了这么久，你还记得我的声音。",
    "欢迎来到今晚的故事时间，让我们一起开始吧。", "一位温柔自信的年轻女性，声音清晰，语气亲切，表达轻快而富有感染力。",
    "繁體中文：臺灣的語言與文化。", "我喜欢用iPhone 15 Pro拍照，效果超棒！", "混合 mixed 文本 text 123。",
    "价格是¥1,280.50元，折扣20%。", "「引号」『书名』（括号）【方括号】《书名号》",
    "日本語のテキストです。カタカナ、ひらがな。", "한국어 문장입니다.", "Ελληνικά Русский текст עברית العربية हिन्दी ไทย",
    "Café naïve résumé coöperate jalapeño Zürich Ångström",
    "é café (decomposed)", "à́̂̃ stacked", "́ leading combining mark",
    "Z͓͑͒a͔l͕g͖o", "กำ thai", "क्ष devanagari conjunct",
    "🚀 Launch day! 🎉🎉 Thanks 🙏🏽 to the team 👩‍👩‍👧‍👦 ❤️ #AI", "\U0001F600" * 5, "🇹🇼🇺🇸🏳️‍🌈", "👍🏿👍🏻",
    "​‌‍ zero width", "﻿BOM start", "Control\x00\x01\x1f chars\x7f", "nbsp ideo　narrow ",
    " private use", "\U0010fffd\U000f0000 planes", "͸͹ unassigned", "￾￿ noncharacters",
    "\U0001d400\U0001d401 math bold", "\U00020000\U0002a6d6 extension B", "𠀀𪛖𫝀", "   separators",
    "▁", "▁▁", "▁" * 31, "▁" * 32, "▁" * 40, "a▁b", "▁leading", "trailing▁", "▁ ▁ ▁", " ▁", "▁ ",
    "\n" * 31, "\n" * 32, "\n" * 40, "\t" * 31, "\t" * 32, "\t" * 40, " " * 31, " " * 32, " " * 33,
    "\n\t\n\t", "\t\n", "\n ", " \n", "\n▁", "▁\n",
    "(laugh) That was funny.", "(sigh) It is good to hear your voice again after all this time.",
    "(clears throat) We need to discuss what happened last night.", "(cough) Sorry. (laugh)",
    "[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。", "[叹气] 没想到过了这么久，你还记得我的声音。", "[咳嗽][清嗓子]好的。",
    "(laugh)(laugh)(laugh)", "[笑][笑]", "( laugh )", "[ 笑 ]", "(Laugh)", "[S0](laugh)", "[S0][笑]",
    "[S0][S1][S2][S3][S4][S5][S6][S7][S8][S9]", "[S0]", "[S1]", "[S9]", "[S10]", "[s0]", "[S0]Hello world.", "[S0][S1][S2]", "[S0] [S1]", "[S 0]", "[S0",
    "S0]", "[[S0]]", "<ins_bos>", "<ins_eos>", "<ins_bos>Speak warmly.<ins_eos>Hello.", "<ins_bos ><ins_eos>",
    "<|AUDIO|>", "<|audio_eos|>", "<|AUDIO|><|AUDIO|><|audio_eos|>", "<|AUDIO|>x<|audio_eos|>", "<| AUDIO |>",
    "<|AUDIO", "AUDIO|>", "<|audio|>", "< bos>", "<bos >", "<b os>", "<bos>", "<bos><bos>", "x<bos>y", "<eos>",
    "<pad><unk><mask>", "<unused0>", "<unused1>", "<unused10>", "<unused100>", "<unused1><unused10>", "<unused",
    "<b>bold</b>", "<bbox>", "<s>strike</s>", "<strong>", "<b><bos><bbox>", "<s><strong></s>", "<table><tr><td>1</td></tr></table>",
    "<start_of_turn>user\nhi<end_of_turn>", "<start_of_image>", "<image_soft_token>", "[multimodal]", "<", ">", "<>", "<<>>",
    "< S0 >", "[ S0 ]", "[S0 ]", "[ S0]", "<ins_ bos>", "<ins_bos", "ins_bos>", "<|AU DIO|>",
    "\\n literal backslash n", "\"quoted json\"", "{\"key\": [1, 2, 3]}", "def f(x):\n    return x * 2\n",
    "SELECT * FROM t WHERE a = 'b';", "https://x.com/a_(b)", "#hashtag @mention", "**bold** _italic_ `code`",
]
pool = (
    "the of and to in is was that for on with as by it at from this be are have an not or which "
    "Breeze voice speech model clone design direction tone pace emotion warm calm serious slow fast "
    "Hello world . , ! ? ; : ' \" - — – … ( ) [ ] { } < > / \\ | @ # $ % ^ & * _ + = ~ ` "
    "我 你 他 的 是 了 在 有 一 不 人 这 中 大 为 上 个 国 们 到 说 时 要 就 出 会 可 也 你们 今天 天气 很好 声音 "
    "。 ， ！ ？ ； ： 、 「 」 （ ） 【 】 《 》 "
    "0 1 2 3 9 10 42 100 2026 3.14 1,000 50% $5 ¥8 "
    "😀 🎉 👩‍👩‍👧‍👦 🙏🏽 ❤️ 🇹🇼 é é ñ ü ß Ø æ œ ç ı İ ﬁ ﬀ Ω µ ° ± × ÷ √ ∞ ≠ ≤ "
    " \U0010fffd ͸ \U0001d400 𠀀 ​ ﻿ \x00 \x7f   　 "
    "(laugh) (cough) (sigh) (clears throat) [笑] [咳嗽] [清嗓子] [叹气]"
).split(" ") + specials + ["▁", "▁▁", "\n", "\n\n", "\t", "\r", " ", "  ", "   "]
for _ in range(420):
    pieces = [rng.choice(pool) for _ in range(rng.randint(1, 40))]
    sep = rng.choice(["", " ", " ", " ", "  ", "\n"])
    corpus.append(sep.join(pieces))
for _ in range(40):
    corpus.append("".join(chr(rng.choice([rng.randint(0x20, 0x7e), rng.randint(0xa0, 0x2fff), rng.randint(0x3000, 0x9fff),
                                         rng.randint(0xac00, 0xd7a3), rng.randint(0xe000, 0xffff), rng.randint(0x10000, 0x10ffff)]))
                      for _ in range(rng.randint(1, 60))))
corpus.append(" ".join(c for c in corpus[:120] if c))
seen, unique = set(), []
for s in corpus:
    if s not in seen:
        seen.add(s)
        unique.append(s)
corpus = unique

tokenizer_cases = []
for s in corpus:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert ids == raw.encode(s, add_special_tokens=False).ids, f"tokenizers disagrees on {s!r}"
    assert tok(s, add_special_tokens=True)["input_ids"] == [tok.bos_token_id] + ids, f"bos on {s!r}"
    tokenizer_cases.append({"text": s, "ids": ids})

# Per-segment versus joint: the reference renders each text segment as decode(encode(text)) and
# re-encodes the joined prompt; a port encodes each segment once. The decoder turns every U+2581 into
# a space, so a port must too: a literal "▁▁" is an added token, but its decoded "  " is not. Every
# corpus string is tried as text, transcript and instruction, on both sides of an audio segment.
for s in corpus:
    assert tok.decode(tok(s)["input_ids"], skip_special_tokens=False) == "<bos>" + s.replace("\u2581", " "), repr(s)
def joint(segments):
    codes_by_path = {}
    def encode_audio(_, path):
        return codes_by_path[path]
    templates._encode_prompt_audio = encode_audio
    rendered = []
    for seg in segments:
        if seg["type"] == "audio":
            codes_by_path[seg["audio_path"]] = torch.zeros((seg["frames"], 16), dtype=torch.int16)
    out = templates._prepare_one(tok, None, SimpleNamespace(num_codebooks=16), segments)
    return out["input_ids"][0].tolist(), out["text_ids_len"].tolist()

def separate(segments):
    ids, lens = [], []
    for seg in segments:
        if seg["type"] == "text":
            seg_ids = tok(seg["text"].replace("\u2581", " "), add_special_tokens=True)["input_ids"]
            ids += seg_ids
            lens.append(len(seg_ids))
        else:
            ids += [262144] * seg["frames"] + [262145]
    return ids, lens

counterexamples = []
for i, s in enumerate(corpus):
    shapes = [
        [{"type": "text", "text": f"[S0]{s}"}],
        [{"type": "text", "text": f"[S0]{s}"}, {"type": "audio", "audio_path": "a", "frames": 3, "append_eos": True},
         {"type": "text", "text": f"[S0]{s}"}],
        [{"type": "text", "text": f"[S0]<ins_bos>{s}<ins_eos>{s}"}],
        [{"type": "text", "text": s}, {"type": "audio", "audio_path": "a", "frames": 1, "append_eos": True}, {"type": "text", "text": s}],
    ]
    for segments in shapes:
        if joint(segments) != separate(segments):
            counterexamples.append(s)
            break
print(f"per-segment vs joint: {len(counterexamples)} counterexamples over {len(corpus)} strings x 4 layouts")
for s in counterexamples[:20]:
    print("  ", repr(s)[:120])

# Prompt fixtures from templates.prepare_inputs. Codes are random in 0..2047, which is all the
# prompt builder looks at.
def ref_codes(frames):
    return [[rng.randrange(2048) for _ in range(16)] for _ in range(frames)]

texts = ["Hello world.", "(sigh) It is good to hear your voice again after all this time.", "[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。",
         "  leading spaces kept", "line\nbreak and\ttab", "混合 mixed 文本 (laugh) 123。", "a" * 300, "a ▁▁ b ▁"]
instructions = ["Speak slowly with a restrained, serious tone.", "一位温柔自信的年轻女性，声音清晰。", "  padded instruction  "]
transcripts = ["This is the exact transcript of the reference audio.", "这是中文参考音频的准确文字稿。", "  padded transcript\n"]

def prompt_case(task, text, instruction=None, transcript=None, frames=None, guidance=1.0):
    request = {"id": "x", "text": text, "speaker": "S0"}
    codes = None
    if instruction is not None:
        request["instruction"] = instruction.strip()
    if transcript is not None:
        codes = ref_codes(frames)
        request["ref_audio_path"] = "ref.wav"
        request["ref_text"] = transcript.strip()
        templates._encode_prompt_audio = lambda _, path: torch.tensor(codes, dtype=torch.int16)
    model = SimpleNamespace(config=SimpleNamespace(num_codebooks=16), device="cpu")
    case = {"task": task, "text": text, "instruction": instruction, "transcript": transcript, "codes": codes, "guidance": guidance}
    try:
        inputs = templates.prepare_inputs(tok, None, model, [request], templates.get_template(templates.select_template_name(request)),
                                          guidance_scale=guidance, guidance_scale_ref=None, guidance_scale_ins=None)
    except ValueError as e:
        case["error"] = str(e)
        return case
    def branch(ids, mask, lengths):
        return {"ids": inputs[ids][0].tolist(), "text_mask": inputs[mask][0].tolist(), "text_lengths": inputs[lengths].tolist()}
    case["positive"] = branch("input_ids", "text_ids_mask", "text_ids_len")
    if "cfg_negative_prompt_ids" in inputs:
        case["negative"] = branch("cfg_negative_prompt_ids", "cfg_negative_text_ids_mask", "cfg_negative_text_ids_len")
    return case

prompt_cases = []
for i, text in enumerate(texts):
    instruction, transcript = instructions[i % 3], transcripts[i % 3]
    frames = [1, 2, 7, 40][i % 4]
    prompt_cases += [
        prompt_case("plain", text),
        prompt_case("voiceDesign", text, instruction=instruction, guidance=4.0),
        prompt_case("voiceDesign", text, instruction=instruction, guidance=1.0),
        prompt_case("voiceClone", text, transcript=transcript, frames=frames),
        prompt_case("voiceDirection", text, instruction=instruction, transcript=transcript, frames=frames, guidance=4.0),
        prompt_case("voiceDirection", text, instruction=instruction, transcript=transcript, frames=frames, guidance=2.5),
    ]
prompt_cases += [prompt_case("plain", "Hello.", guidance=4.0), prompt_case("voiceClone", "Hello.", transcript="Hi.", frames=3, guidance=4.0)]
assert sum("error" in c for c in prompt_cases) == 2

with open(os.path.join(here, "tokenizer_fixtures.json"), "w") as f:
    json.dump({"cases": tokenizer_cases}, f, ensure_ascii=False)
with open(os.path.join(here, "prompt_fixtures.json"), "w") as f:
    json.dump({"cases": prompt_cases}, f, ensure_ascii=False)
print(f"{len(tokenizer_cases)} tokenizer cases, {len(prompt_cases)} prompt cases")
assert not counterexamples
