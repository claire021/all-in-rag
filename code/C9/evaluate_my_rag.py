import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from ragas import evaluate

# 1. 导入新版指标
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.llms import llm_factory
from ragas.run_config import RunConfig  
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings.base import BaseRagasEmbeddings

# 导入你的系统核心类
from main import AdvancedGraphRAGSystem

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False


class RagasHuggingFaceEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model_name: str):
        self.langchain_embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.langchain_embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.langchain_embeddings.embed_documents(texts)

    async def embed_text(self, text: str) -> list[float]:
        return self.langchain_embeddings.embed_query(text)

    async def aembed_query(self, text: str) -> list[float]:
        return self.langchain_embeddings.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.langchain_embeddings.embed_documents(texts)


def build_stable_test_dataset():
    print("\n========= 1. 载入标准测试题库 =========")
    test_data = {
        'question': [
            "中式馅饼的准备时间需要多久？",
            "如何制作韭菜盒子需要哪些食材和步骤？",
            "甜辣烤全翅的准备时间是多少分钟？"
        ],
        'ground_truth': [
            "中式馅饼的准备时间约为35分钟到40分钟（含醒面时间）。",
            "食材包括韭菜、鸡蛋、虾仁、面粉、盐等。步骤包括和面醒发、切配调馅、包制成形以及小火煎至两面金黄。",
            "甜辣烤全翅的准备时间是15分钟。"
        ]
    }
    return pd.DataFrame(test_data)


def run_rag_pipeline_and_collect(rag_system, df_test):
    print("\n========= 2. 运行图RAG系统收集回答 =========")
    questions = df_test['question'].tolist()
    ground_truths = df_test['ground_truth'].tolist()
    
    final_questions, final_answers, final_contexts, final_ground_truths = [], [], [], []
    
    for i, q in enumerate(questions):
        print(f"正在测试第 [{i+1}/{len(questions)}] 题: {q}")
        relevant_docs, _ = rag_system.query_router.route_query(q, rag_system.config.top_k)
        contexts = [doc.page_content for doc in relevant_docs] if relevant_docs else ["未检索到相关本地知识库内容。"]
        answer = rag_system.generation_module.generate_adaptive_answer(q, relevant_docs)
        
        final_questions.append(q)
        final_answers.append(answer)
        final_contexts.append(contexts)
        final_ground_truths.append(ground_truths[i])
        
    eval_dict = {
        "question": final_questions,
        "contexts": final_contexts,
        "answer": final_answers,
        "ground_truth": final_ground_truths
    }
    return Dataset.from_dict(eval_dict)


def start_evaluation_and_plot(rag_system, eval_dataset):
    print("\n========= 3. 调用 RAGAS 启动自动化打分 =========")
    
    ragas_judge_llm = llm_factory(
        model=rag_system.config.llm_model,
        client=rag_system.generation_module.client
    )
    
    ragas_embeddings = RagasHuggingFaceEmbeddings(model_name=rag_system.config.embedding_model)
    
    m1 = Faithfulness()
    m2 = AnswerRelevancy()
    m3 = AnswerCorrectness()
    
    m1.llm = ragas_judge_llm
    m2.llm = ragas_judge_llm
    m2.embeddings = ragas_embeddings
    m3.llm = ragas_judge_llm
    m3.embeddings = ragas_embeddings

    safe_eval_config = RunConfig(timeout=60, max_workers=1, max_retries=5, max_wait=5)
    
    # 核心步骤：Ragas计算
    result = evaluate(
        dataset=eval_dataset,
        metrics=[m1, m2, m3],
        llm=ragas_judge_llm,
        embeddings=ragas_embeddings,
        run_config=safe_eval_config
    )
    
    print("\n📊 【最终正式评测报告分数】")
    print(result)
    
    print("\n📈 正在为您绘制可视化评测报告图...")
    
    # ================= 🚨 深度防御检查区域 🚨 =================
    # 1. 放弃所有 Ragas 专属对象的类取值，降维成原生 pandas DataFrame
    df_result = result.to_pandas()
    
    # 2. 导出 CSV 备份，这里的表头会被 pandas 自动对齐为原生列名
    df_result.to_csv("my_ragas_report.csv", index=False)
    
    # 3. 提取平均分（防御 nan 和缺失列）
    # 用 .mean() 自动聚合多行测试题的平均分。如果全为 nan，pandas 会返回 nan
    # 为了防止 matplotlib 无法绘制 nan，引入 float() 并在最后一步用 .fillna(0) 兜底
    scores = {
        '忠实度 (Faithfulness)': float(df_result['faithfulness'].fillna(0).mean()) if 'faithfulness' in df_result.columns else 0.0,
        '回答相关性 (Answer Relevancy)': float(df_result['answer_relevancy'].fillna(0).mean()) if 'answer_relevancy' in df_result.columns else 0.0,
        '综合正确性 (Answer Correctness)': float(df_result['answer_correctness'].fillna(0).mean()) if 'answer_correctness' in df_result.columns else 0.0
    }
    
    # 4. 绘图检查
    plt.figure(figsize=(8, 5))
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    # 显式转换 keys 和 values 为标准 list，防止 matplotlib 无法解析 dict_keys 类型
    x_labels = list(scores.keys())
    y_values = list(scores.values())
    
    bars = plt.bar(x_labels, y_values, color=colors, width=0.4)
    plt.ylim(0, 1.1)
    plt.title('GraphRAG 系统多维度自动化评测报告', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('得分 (0.0 - 1.0)', fontsize=12)
    
    # 5. 数值标签渲染检查
    for bar in bars:
        height = bar.get_height()
        # 确保分数值以四位小数完美对齐
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                 
    plt.tight_layout()
    plt.savefig("rag_eval_report.png", dpi=300)
    plt.close() # 显式释放内存，防止文件句柄死锁
    print("✅ 完美的图表报告已保存至当前目录下的: rag_eval_report.png")
if __name__ == "__main__":
    rag_system = AdvancedGraphRAGSystem()
    rag_system.initialize_system()
    rag_system.build_knowledge_base() 
    
    try:
        df_test = build_stable_test_dataset()
        eval_dataset = run_rag_pipeline_and_collect(rag_system, df_test)
        start_evaluation_and_plot(rag_system, eval_dataset)
        
    except Exception as e:
        print(f"\n❌ 运行过程中发生错误: {e}")
        raise e
    finally:
        print("\n正在优雅关闭系统资源...")
        if hasattr(rag_system, '_cleanup'):
            rag_system._cleanup()