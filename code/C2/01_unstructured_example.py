from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from collections import Counter
# PDF文件路径
pdf_path = "././data/C2/pdf/rag.pdf"

# hi_res 模式
print("=== hi_res 模式 ===")
elements_hi = partition_pdf(filename=pdf_path, strategy="hi_res")
print(f"元素数量: {len(elements_hi)}")
print(f"元素类型: {dict(Counter(e.category for e in elements_hi))}")
print("\n前3个元素:")
for i, e in enumerate(elements_hi[:3], 1):
    print(f"{i}. [{e.category}] {str(e)[:100]}")

print("\n" + "="*60 + "\n")

# ocr_only 模式
print("=== ocr_only 模式 ===")
elements_ocr = partition_pdf(filename=pdf_path, strategy="ocr_only")
print(f"元素数量: {len(elements_ocr)}")
print(f"元素类型: {dict(Counter(e.category for e in elements_ocr))}")
print("\n前3个元素:")
for i, e in enumerate(elements_ocr[:3], 1):
    print(f"{i}. [{e.category}] {str(e)[:100]}")

# -------------使用 Unstructured 库加载并解析一个PDF文件——-----------

# 使用Unstructured加载并解析PDF文档
elements = partition(
    filename=pdf_path,
    content_type="application/pdf"
)

# 打印解析结果
print(f"解析完成: {len(elements)} 个元素, {sum(len(str(e)) for e in elements)} 字符")

# 统计元素类型
types = Counter(e.category for e in elements)
print(f"元素类型: {dict(types)}")

# 显示所有元素
print("\n所有元素:")
for i, element in enumerate(elements, 1):
    print(f"Element {i} ({element.category}):")
    print(element)
    print("=" * 60)
