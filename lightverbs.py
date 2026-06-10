# pip install torch transformers pandas numpy scikit-learn matplotlib

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from matplotlib import font_manager as fm
from bidi.algorithm import get_display
import arabic_reshaper
URDU_FONT_PATH = "./NotoSansArabic-Regular.ttf"
try:
    urdu_font = fm.FontProperties(fname=URDU_FONT_PATH)
    print("Using font:", urdu_font.get_name())
except Exception as e:
    urdu_font = None
    print("Font not loaded, continuing without Urdu font. Reason:", e)
def reshape_urdu(text):
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

MODEL_NAME = "../UrduBert/urdu-bert-64k-17epochs"

#MODEL_NAME = "bert-base-multilingual-cased"

# ----------------------------
# 1. Crafted Urdu mini-corpus
# ----------------------------

examples = {
    "دیا": {
        "main": [
            "اس نے مجھے قلم دیا",
            "استاد نے طالب علم کو جواب دیا",
            "ماں نے بچے کو دودھ دیا",
            "دوست نے مجھے تحفہ دیا",
            "حکومت نے عوام کو ریلیف دیا",
            "ڈاکٹر نے مریض کو مشورہ دیا",
            "اس نے فقیر کو پیسہ دیا",
            "لڑکے نے کتاب واپس دی",
            "والد نے بیٹے کو اجازت دی",
            "اس نے مجھے اپنا نمبر دیا",
        ],
        "light": [
            "اس نے کام ختم کر دیا",
            "بچے نے دروازہ بند کر دیا",
            "میں نے خط لکھ دیا",
            "اس نے مسئلہ حل کر دیا",
            "انہوں نے فیصلہ سنا دیا",
            "لڑکی نے سبق یاد کر دیا",
            "مزدور نے گھر بنا دیا",
            "اس نے کپڑے دھو دیے",
            "میں نے سب کچھ بتا دیا",
            "اس نے چراغ جلا دیا",
        ],
    },

    "گیا": {
        "main": [
            "وہ کل لاہور گیا",
            "احمد بازار گیا",
            "بچہ اسکول گیا",
            "میرا بھائی دفتر گیا",
            "وہ ڈاکٹر کے پاس گیا",
            "استاد کلاس میں گیا",
            "مسافر اسٹیشن گیا",
            "وہ مسجد گیا",
            "دوست میرے گھر گیا",
            "لڑکا میدان میں گیا",
        ],
        "light": [
            "کام مکمل ہو گیا",
            "دروازہ بند ہو گیا",
            "مسئلہ حل ہو گیا",
            "بچہ سو گیا",
            "کھیل ختم ہو گیا",
            "بارش شروع ہو گئی",
            "چراغ بجھ گیا",
            "دل ٹوٹ گیا",
            "رنگ بدل گیا",
            "معاملہ واضح ہو گیا",
        ],
    },

    "بیٹھا": {
        "main": [
            "وہ کرسی پر بیٹھا",
            "بچہ زمین پر بیٹھا",
            "استاد میز کے پاس بیٹھا",
            "احمد خاموشی سے بیٹھا",
            "مہمان صوفے پر بیٹھا",
            "لڑکا درخت کے نیچے بیٹھا",
            "وہ کمرے میں بیٹھا",
            "بوڑھا آدمی بینچ پر بیٹھا",
            "شاگرد کلاس میں بیٹھا",
            "مسافر بس میں بیٹھا",
        ],
        "light": [
            "وہ اچانک رو بیٹھا",
            "بچہ بات کہہ بیٹھا",
            "میں غلطی کر بیٹھا",
            "وہ راز بتا بیٹھا",
            "لڑکا وعدہ کر بیٹھا",
            "وہ ہنس بیٹھا",
            "احمد ناراض ہو بیٹھا",
            "وہ انکار کر بیٹھا",
            "میں سوال پوچھ بیٹھا",
            "وہ فیصلہ کر بیٹھا",
        ],
    },

    "اٹھا": {
        "main": [
            "وہ صبح جلدی اٹھا",
            "بچہ نیند سے اٹھا",
            "احمد کرسی سے اٹھا",
            "وہ زمین سے اٹھا",
            "مریض بستر سے اٹھا",
            "لڑکا نماز کے لیے اٹھا",
            "وہ اچانک اٹھا",
            "بوڑھا آدمی آہستہ اٹھا",
            "مہمان کھانے کے بعد اٹھا",
            "استاد میز سے اٹھا",
        ],
        "light": [
            "وہ زور سے بول اٹھا",
            "بچہ اچانک ہنس اٹھا",
            "مجمع شور مچا اٹھا",
            "وہ خوشی سے چلا اٹھا",
            "عورت رو اٹھا",
            "لڑکا چیخ اٹھا",
            "دل تڑپ اٹھا",
            "لوگ احتجاج کر اٹھے",
            "احمد سوال کر اٹھا",
            "سامعین داد دے اٹھے",
        ],
    },

    "پڑا": {
        "main": [
            "کتاب میز پر پڑی تھی",
            "کپڑا زمین پر پڑا تھا",
            "خط دراز میں پڑا تھا",
            "بیگ کمرے میں پڑا تھا",
            "پتھر راستے میں پڑا تھا",
            "جوتا دروازے کے پاس پڑا تھا",
            "اخبار صوفے پر پڑا تھا",
            "کھلونا فرش پر پڑا تھا",
            "پرس گاڑی میں پڑا تھا",
            "موبائل بستر پر پڑا تھا",
        ],
        "light": [
            "وہ اچانک ہنس پڑا",
            "بچہ زور سے رو پڑا",
            "احمد غصے میں بول پڑا",
            "لڑکی خوف سے چیخ پڑی",
            "وہ بات سن کر مسکرا پڑا",
            "استاد حیران رہ پڑا",
            "میں جواب دے پڑا",
            "وہ بے اختیار کہہ پڑا",
            "لوگ تالیاں بجا پڑے",
            "بچہ سوال پوچھ پڑا",
        ],
    },

    "آیا": {
        "main": [
            "وہ کل گھر آیا",
            "احمد بازار  آیا",
            "مہمان شام کو آیا",
            "بچہ اسکول آیا",
            "ڈاکٹر اسپتال آیا",
            "استاد کلاس میں آیا",
            "دوست میرے پاس آیا",
            "بھائی لاہور آیا",
            "مسافر اسٹیشن آیا",
            "وہ دیر سے آیا",
        ],
        "light": [
           "اس کے ذہن میں تصویرکشی کا انوکھا مصرف ابھر آیا",
            "اقرار الحسن کے جلسے میں لوگوں کا ٹھاٹھیں مارتا ہوا سمندر امنڈ آیا",
            "بھالو والا ڈگڈگی بجاتا آیا",
            "ایک بندہ بھاگتا آیا",
            "وہ بولتا آیا",
            "کسان کے جاتے ہی ایک گڈریا چشمے پر اپنی بکریوں کو پانی پلانے آیا",
            "میں اس کے سارے خط جلا آیا",
            "کوئی کندھوں پہ چڑھ آیا",
            "وہ سرکس دیکھنے چلا آیا",
            "اور وہ تیرے پاس دوڑتا آیا",
            "ایک نابینا شیش محل دیکھنے آیا",
            "زخمی طالب علم ایمبولینس میں نہم کا پرچہ دینے آیا",
            "مجھے درد سے رونا آیا",
            "لیکن اسے ادھورا چھوڑ کر ایک نئی زندگی کی طرف کیوں لوٹ آیا",
            "ایک دن مالک مکان کرایہ لینے آیا",
            "بچہ کلاس میں اپنا کھلونا لے آیا",
            "ایک کافر شخص آپ کو مارنے آیا",
            "مجھے افق پر عید کا چاند نظر آیا",
            "نیز وہ منی لانڈرنگ والی فہرست سے بھی نکل آیا",
            "تیسرے نمبر پہ ایک حرامی ہنستا آیا", 
        ],
    },
}

rows = []
for verb, groups in examples.items():
    for usage, sents in groups.items():
        for sent in sents:
            rows.append({"verb": verb, "usage": usage, "sentence": sent})

df = pd.DataFrame(rows)

# ----------------------------
# 2. Load mBERT
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ----------------------------
# 3. Extract target embedding
# ----------------------------

def get_target_embedding(sentence, target):
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=128
    )

    offsets = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)

    hidden = outputs.hidden_states[-1][0]

    start = sentence.find(target)
    if start == -1:
        return None

    end = start + len(target)

    token_indices = []
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if max(s, start) < min(e, end):
            token_indices.append(i)

    if not token_indices:
        return None

    return hidden[token_indices].mean(dim=0).numpy()

df["embedding"] = df.apply(
    lambda row: get_target_embedding(row["sentence"], row["verb"]),
    axis=1
)

df = df[df["embedding"].notnull()].reset_index(drop=True)

# ----------------------------
# 4. Main vs light comparison
# ----------------------------

results = []

for verb in df["verb"].unique():
    sub = df[df["verb"] == verb]

    main_vecs = np.stack(sub[sub["usage"] == "main"]["embedding"])
    light_vecs = np.stack(sub[sub["usage"] == "light"]["embedding"])

    main_centroid = main_vecs.mean(axis=0)
    light_centroid = light_vecs.mean(axis=0)

    similarity = cosine_similarity([main_centroid], [light_centroid])[0][0]
    distance = 1 - similarity

    results.append({
        "verb": verb,
        "main_examples": len(main_vecs),
        "light_examples": len(light_vecs),
        "cosine_similarity": similarity,
        "cosine_distance": distance
    })

results_df = pd.DataFrame(results).sort_values(
    "cosine_distance",
    ascending=False
)

print("\nMain vs Light Verb Distance")
print(results_df)

results_df.to_csv("urdu_mbert_lightverb_results.csv", index=False)

# ----------------------------
# 5. PCA visualization
# ----------------------------

X = np.stack(df["embedding"].values)

pca = PCA(n_components=2)
coords = pca.fit_transform(X)

df["pc1"] = coords[:, 0]
df["pc2"] = coords[:, 1]

plt.figure(figsize=(10, 7))

for usage in ["main", "light"]:
    sub = df[df["usage"] == usage]
    
    plt.scatter(sub["pc1"], sub["pc2"], label=usage, alpha=0.7)

for _, row in df.iterrows():
    plt.text(row["pc1"], row["pc2"], reshape_urdu(row["verb"]), fontsize=8)

plt.title("mBERT Embeddings: Urdu Main Verb vs Light Verb Uses")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig("urdu_mbert_lightverb_pca.png", dpi=300)
plt.show()
