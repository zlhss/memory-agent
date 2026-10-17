from sklearn.metrics.pairwise import cosine_similarity

from memory_search import model, memories, memory_embeddings
from ollama_client import filter_relevant_memories

test_cases = [
    # 应该召回
    {
        "query": "我喜欢听什么？",
        "expected": "用户喜欢听丸之内虐待狂"
    },
    {
        "query": "我平时喜欢听什么歌？",
        "expected": "用户喜欢听丸之内虐待狂"
    },
    {
        "query": "我在学什么语言？",
        "expected": "用户正在学习日语"
    },
    {
        "query": "我最近在学哪门语言？",
        "expected": "用户正在学习日语"
    },
    {
        "query": "我住在哪里？",
        "expected": "用户住在上海"
    },
    {
        "query": "我现在住哪座城市？",
        "expected": "用户住在上海"
    },
    {
        "query": "我妹妹叫什么？",
        "expected": "用户妹妹叫 Alice"
    },
    {
        "query": "Alice 和我是什么关系？",
        "expected": "用户妹妹叫 Alice"
    },
    {
        "query": "Alice 喜欢什么运动？",
        "expected": "Alice 喜欢跑步"
    },
    {
        "query": "我喜欢吃什么？",
        "expected": "用户喜欢吃苹果"
    },

    # 不应该召回任何长期记忆
    {
        "query": "量子力学是什么？",
        "expected": None
    },
    {
        "query": "法国的首都是哪里？",
        "expected": None
    },
    {
        "query": "帮我写一个冒泡排序",
        "expected": None
    },
    {
        "query": "Python 的字典是什么？",
        "expected": None
    },
    {
        "query": "太阳有多大？",
        "expected": None
    },
    {
        "query": "怎么计算圆的面积？",
        "expected": None
    },
    {
        "query": "Transformer 是什么？",
        "expected": None
    },
    {
        "query": "今天星期几？",
        "expected": None
    },
    {
        "query": "二战什么时候结束？",
        "expected": None
    },
    {
        "query": "什么是梯度下降？",
        "expected": None
    },
]

def raw_retrieve(query):
    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        memory_embeddings
    )[0]

    pairs = list(zip(similarities, memories))

    ranked = sorted(
        pairs,
        key=lambda x: x[0],
        reverse=True
    )

    return ranked

eval_results = []
threshold = 0.14

for case in test_cases:
    query = case["query"]
    expected = case["expected"]
    
    ranked = raw_retrieve(query)
    
    top_score, top_memory = ranked[0]
    


    candidate_memories = []

    for score, memory in ranked[:3]:
        if score >= threshold:
            candidate_memories.append(memory)
            
    v2_memories = filter_relevant_memories(
    query,
    candidate_memories
)
    
    eval_results.append({
    "query": query,
    "expected": expected,
    "top_score": float(top_score),
    "top_memory": top_memory,
    "v1_memories": candidate_memories,
    "v2_memories": v2_memories
})



def evaluate_threshold(threshold):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    for result in eval_results:
        expected = result["expected"]
        top_score = result["top_score"]
        
        should_retrieve = expected is not None
        predicted_retrieve = top_score >= threshold
        
        if should_retrieve and predicted_retrieve:
            tp += 1

        elif not should_retrieve and predicted_retrieve:
            fp += 1

        elif should_retrieve and not predicted_retrieve:
            fn += 1

        else:
            tn += 1
            
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    precision = tp / (tp + fp) if tp + fp > 0 else 0

    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )
    
    return {
    "threshold": threshold,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "tn": tn
}       

def evaluate_method(method_key):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for result in eval_results:
        expected = result["expected"]
        retrieved = result[method_key]

        # 本来不应该召回任何记忆
        if expected is None:
            if len(retrieved) == 0:
                tn += 1
            else:
                fp += 1

        # 本来应该召回某条记忆
        else:
            if expected in retrieved:
                tp += 1
            else:
                fn += 1

    accuracy = (tp + tn) / (tp + fp + fn + tn)

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }
    
    
v1_result = evaluate_method("v1_memories")
v2_result = evaluate_method("v2_memories")

for result in eval_results:
    print("\nQuery:", result["query"])
    print("Expected:", result["expected"])
    print("V1:", result["v1_memories"])
    print("V2:", result["v2_memories"])

print("\nV1:")
print(v1_result)

print("\nV2:")
print(v2_result)