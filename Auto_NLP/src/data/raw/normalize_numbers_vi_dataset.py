# -*- coding: utf-8 -*-
"""
Chuẩn hóa toàn bộ dataset:
- Convert tất cả số -> chữ tiếng Việt (kể cả trong entities/values), KHÔNG TRỪ field nào.
- Quy tắc:
  + Số điện thoại: 0-9 đọc từng chữ số ("không chín bảy...").
  + %: "50%" -> "năm mươi phần trăm".
  + Giờ: "7:30", "07:30", "7h30", "7h" -> "bảy giờ ba mươi", "bảy giờ".
  + Ngày/tháng: "24/10", "1-9" -> "hai mươi bốn tháng mười", "một tháng chín".
  + Số còn lại: 1..9999 -> chữ (một, mười, hai mươi mốt, …).
- Sau chuẩn hóa: rebuild lại spans, entities, bio_labels, values (text).
- Không đụng "split", "intent", "command".

Cách chạy:
    python normalize_numbers_vi_dataset.py elderly_commands_master.json
"""

import sys, json, re, unicodedata
from collections import Counter

VN_WORD = r"[0-9A-Za-zÀ-ỹđĐ]"

def nfc(s): return unicodedata.normalize("NFC", s) if isinstance(s, str) else s

# ---------- Number to words (basic, đủ 0..9999) ----------
digits = ["không","một","hai","ba","bốn","năm","sáu","bảy","tám","chín"]

def two_digits(n):
    assert 0 <= n <= 99
    if n < 10: return digits[n]
    if n == 10: return "mười"
    tens, ones = divmod(n, 10)
    base = "mười" if tens == 1 else digits[tens] + " mươi"
    if ones == 0: return base
    if ones == 1: return base + " mốt" if tens >= 2 else base + " một"
    if ones == 4 and tens >= 2: return base + " tư"
    if ones == 5: return base + " lăm"
    return base + " " + digits[ones]

def three_digits(n):
    assert 0 <= n <= 999
    if n < 100: return two_digits(n)
    hundreds, rest = divmod(n, 100)
    head = digits[hundreds] + " trăm"
    if rest == 0: return head
    tens, ones = divmod(rest, 10)
    if tens == 0:
        # x trăm lẻ y
        if ones == 0: return head
        return head + " lẻ " + digits[ones]
    return head + " " + two_digits(rest)

def number_vi(n):
    n = int(n)
    if n < 1000: return three_digits(n)
    if n < 10000:
        thousands, rest = divmod(n, 1000)
        s = (digits[thousands] + " nghìn").strip()
        if rest == 0: return s
        if rest < 100: return s + " không trăm " + two_digits(rest) if rest < 100 else s + " " + three_digits(rest)
        return s + " " + three_digits(rest)
    # fallback: đọc từng chữ số (hiếm khi cần)
    return " ".join(digits[int(ch)] for ch in str(n))

def each_digit(s):
    return " ".join(digits[int(ch)] for ch in s)

# ---------- Regex cho các case ----------
RE_PHONE = re.compile(r"(?<!\d)(?:\+?84|0)\d{7,12}(?!\d)")   # chuỗi số dài -> đọc từng chữ số
RE_PERCENT = re.compile(r"(\d{1,3})\s*%")
RE_TIME_HHMM = re.compile(r"\b([01]?\d|2[0-3])[:h]([0-5]\d)\b")  # 7:30 / 7h30 / 07:30
RE_TIME_HH = re.compile(r"\b([01]?\d|2[0-3])\s*h\b")             # 7h
RE_DATE_DDMM = re.compile(r"\b(3[01]|[12]?\d)\s*[/\-]\s*(1[0-2]|0?[1-9])\b")  # 24/10, 1-9

# số rời rạc (không nằm trong pattern khác) -> chữ; tránh bắt dính vào từ
RE_NUM_STANDALONE = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")

# ---------- Chuẩn hóa text ----------
def norm_text_vi(text):
    t = nfc(text)

    # 1) Phone: đọc từng chữ số
    def sub_phone(m):
        raw = m.group(0)
        # số điện thoại giữ nguyên dấu '+'? quy về chữ số thôi
        num = re.sub(r"\D", "", raw.lstrip("+"))
        return each_digit(num)
    t = RE_PHONE.sub(sub_phone, t)

    # 2) % -> "… phần trăm"
    def sub_percent(m):
        n = int(m.group(1))
        return number_vi(n) + " phần trăm"
    t = RE_PERCENT.sub(sub_percent, t)

    # 3) Giờ: "hh:mm"/"hhhmm" -> "… giờ …"
    def sub_time_hhmm(m):
        hh = int(m.group(1)); mm = int(m.group(2))
        return f"{number_vi(hh)} giờ {number_vi(mm)}"
    t = RE_TIME_HHMM.sub(sub_time_hhmm, t)

    # 4) Giờ: "hhh" -> "… giờ"
    def sub_time_hh(m):
        hh = int(m.group(1))
        return f"{number_vi(hh)} giờ"
    t = RE_TIME_HH.sub(sub_time_hh, t)

    # 5) Ngày/tháng: dd/mm -> "… tháng …"
    def sub_date(m):
        dd = int(m.group(1)); mm = int(m.group(2))
        return f"{number_vi(dd)} tháng {number_vi(mm)}"
    t = RE_DATE_DDMM.sub(sub_date, t)

    # 6) Số rời rạc còn lại 1..9999 -> chữ
    def sub_num(m):
        n = int(m.group(1))
        return number_vi(n)
    t = RE_NUM_STANDALONE.sub(sub_num, t)

    # dọn khoảng trắng dư
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------- BIO helpers ----------
def token_spans(text):
    spans=[]; i=0; n=len(text)
    while i<n:
        while i<n and text[i].isspace(): i+=1
        if i>=n: break
        s=i
        while i<n and not text[i].isspace(): i+=1
        e=i
        spans.append((text[s:e], s, e))
    return spans

def spans_to_bio(text, spans):
    toks = token_spans(text)
    tags = ["O"]*len(toks)
    for sp in spans:
        st,en,lab = sp["start"], sp["end"], sp["label"]
        first=True
        for i,(tok,a,b) in enumerate(toks):
            if b<=st or a>=en: continue
            tags[i] = ("B-" if first else "I-") + lab
            first=False
    return tags

def find_span_ci(text, sub):
    """Tìm span theo biên từ Unicode, không ăn chuỗi con (case-insensitive)."""
    if not text or not sub: return None
    text_n, sub_n = nfc(text), nfc(sub)
    pat = r"(?<!%s)%s(?!%s)" % (VN_WORD, re.escape(sub_n), VN_WORD)
    m = re.search(pat, text_n, flags=re.IGNORECASE)
    if m: return (m.start(), m.end())
    m = re.search(re.escape(sub_n), text_n, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None

# ---------- Chuẩn hóa từng record ----------
def normalize_record(rec):
    # 1) Chuẩn hóa input
    raw_inp = rec.get("input","")
    norm_inp = norm_text_vi(raw_inp)

    # 2) Chuẩn hóa entity texts (theo cùng quy tắc) rồi tìm lại spans trên norm_inp
    raw_entities = rec.get("entities",[]) or []
    new_entities = []
    for e in raw_entities:
        lab = e.get("label","")
        txt = e.get("text","")
        norm_txt = norm_text_vi(txt)
        sp = find_span_ci(norm_inp, norm_txt)
        if sp:
            new_entities.append({
                "label": lab,
                "text": norm_txt,
                "start": sp[0],
                "end": sp[1],
            })
        else:
            # không tìm thấy: bỏ entity này (tránh tạo span sai)
            pass

    # 3) Nếu không có entities, giữ nguyên logic spans rỗng
    new_entities.sort(key=lambda x:(x["start"], x["end"], x["label"]))
    new_spans = [{"start":e["start"],"end":e["end"],"label":e["label"]} for e in new_entities]
    new_bio = spans_to_bio(norm_inp, new_spans)

    # 4) Chuẩn hóa values (nếu có)
    new_values = []
    for v in rec.get("values",[]) or []:
        lab = v.get("label","")
        txt = v.get("text","")
        if isinstance(txt, str):
            new_values.append({"label": lab, "text": norm_text_vi(txt)})
        else:
            new_values.append({"label": lab, "text": txt})

    # 5) Trả record mới
    out = dict(rec)
    out["input"] = norm_inp
    out["entities"] = new_entities
    out["spans"] = new_spans
    out["bio_labels"] = new_bio
    out["values"] = new_values
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python normalize_numbers_vi_dataset.py <INPUT_JSON>")
        sys.exit(1)
    in_path = sys.argv[1]
    if in_path.lower().endswith(".json"):
        out_path = in_path[:-5] + "_VITEXT.json"
        rep_path = in_path[:-5] + "_VITEXT_REPORT.json"
    else:
        out_path = in_path + "_VITEXT.json"
        rep_path = in_path + "_VITEXT_REPORT.json"

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Dataset có thể là list hoặc dict có key "data"
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        items = data["data"]
        wrap = True
    elif isinstance(data, list):
        items = data
        wrap = False
    else:
        print("ERROR: JSON phải là list các record, hoặc {'data':[...]} giống các file bạn đưa.")
        sys.exit(2)

    fixed = []
    stats = Counter()
    dropped_examples = []

    for idx, rec in enumerate(items):
        try:
            new_rec = normalize_record(rec)
            fixed.append(new_rec)
        except Exception as ex:
            stats["error_records"] += 1
            dropped_examples.append({"idx": idx, "input": rec.get("input","")[:160], "err": str(ex)})

    stats["total"] = len(items)
    stats["fixed"] = len(fixed)

    # Gói lại theo format gốc
    out_data = {"data": fixed} if wrap else fixed

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "errors_preview": dropped_examples[:30]}, f, ensure_ascii=False, indent=2)

    print("DONE")
    print("Saved:", out_path)
    print("Report:", rep_path)

if __name__ == "__main__":
    main()
