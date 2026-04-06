#!/usr/bin/env python3
"""
Demo Prediction Script - Dùng cho presentation
"""

import pickle
import numpy as np

# Vietnamese stopwords
STOPWORDS = set([
    'của', 'và', 'các', 'có', 'được', 'trong', 'với', 'cho', 'này', 'để',
    'tại', 'trên', 'từ', 'về', 'là', 'đến', 'như', 'khi', 'cũng', 'nhưng',
    'đã', 'đang', 'sẽ', 'mà', 'thì', 'nên', 'vẫn', 'rất', 'nhiều', 'hơn',
    'khác', 'phải', 'nếu', 'hay', 'hoặc', 'nhất', 'mỗi', 'ngay', 'ra', 'vào',
    'lại', 'đây', 'kia', 'đó', 'những', 'còn', 'không', 'trong', 'ngoài',
])

THRESHOLD = 0.48


def load_model():
    with open("results/best_model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("results/best_model/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def preprocess(text):
    """Tiền xử lý văn bản tiếng Việt"""
    # Lowercase và loại stopwords
    words = text.lower().split()
    filtered = [w for w in words if w not in STOPWORDS]
    return ' '.join(filtered)


def predict(text, model, vectorizer):
    """Dự đoán sentiment cho một văn bản"""
    processed = preprocess(text)
    vec = vectorizer.transform([processed]).toarray()
    proba = model.predict_proba(vec)[0, 1]
    label = "POSITIVE" if proba >= THRESHOLD else "NON-POSITIVE"
    return label, proba


def main():
    print("=" * 70)
    print("🚀 VIETNAMESE FINANCIAL SENTIMENT ANALYSIS - DEMO")
    print("=" * 70)
    print(f"Model: XGBoost | Accuracy: 84.4% | Threshold: {THRESHOLD}")
    print("-" * 70)

    # Load model
    print("\n📦 Loading model...")
    model, vectorizer = load_model()
    print("✅ Model loaded successfully!\n")

    # Demo samples
    test_samples = [
        "Vingroup báo cáo lợi nhuận kỷ lục, tăng trưởng 200% so với năm ngoái",
        "Thị trường chứng khoán Việt Nam sụt giảm mạnh lo ngại lãi suất Mỹ tăng",
        "Ngân hàng Nhà nước giữ nguyên lãi suất điều hành",
        "Công ty cổ phần A công bố khoản lỗ lớn trong quý 4",
        "Chứng khoán Việt Nam tăng điểm phiên thứ 5 liên tiếp",
        "Vàng giảm giá mạnh nhất trong 3 tháng",
        "Bất động sản ven đô TP.HCM vẫn hấp dẫn nhà đầu tư",
        "Dự báo kinh tế Việt Nam năm 2025 tăng trưởng 7%",
    ]

    print("📊 TESTING WITH SAMPLE NEWS:")
    print("-" * 70)

    for i, text in enumerate(test_samples, 1):
        label, proba = predict(text, model, vectorizer)

        # Visual indicator
        if label == "POSITIVE":
            indicator = "📈"
        else:
            indicator = "📉"

        print(f"\n[{i}] {text[:60]}...")
        print(f"    {indicator} Prediction: {label}")
        print(f"    📊 Confidence: {proba:.1%}")

    print("\n" + "=" * 70)
    print("🎯 INTERACTIVE MODE - Nhập tin tức để test (gõ 'quit' để thoát)")
    print("=" * 70)

    while True:
        try:
            text = input("\n📝 Nhập tin tức: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not text:
                continue

            label, proba = predict(text, model, vectorizer)
            indicator = "📈" if label == "POSITIVE" else "📉"
            print(f"   {indicator} Prediction: {label}")
            print(f"   📊 Confidence: {proba:.1%}")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
