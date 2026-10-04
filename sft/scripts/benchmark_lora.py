import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from swift import (
    InferRequest,
    InferStats,
    RequestConfig,
    TransformersEngine,
    get_processor,
    get_template,
)


DEFAULT_QUESTIONS = [
    """
患者女性，42岁，近一个月容易疲倦，食欲下降，
饭后腹胀，大便偏溏，舌淡苔白。
请从中医辨证角度进行分析，并说明判断依据。
不需要给出具体药物剂量。
""".strip(),

    """
请介绍四君子汤的传统组成、主要功效和经典适应证。
这里只进行中医方剂知识说明，不针对具体患者开方。
""".strip(),

    """
患者只说“最近经常头晕”，目前没有提供更多资料。
作为问诊助手，你下一步应该重点询问哪些信息？
""".strip(),

    """
请用简单的语言解释哈希表的工作原理，
并说明发生哈希冲突时常见的两种处理方法。
""".strip(),
]


def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def create_engine(
    model_path: str,
    adapter_path: str | None,
):
    processor = get_processor(model_path)

    # 显式关闭 thinking，
    # 与本次 non-CoT SFT 的测试目标保持一致
    template = get_template(
        processor,
        enable_thinking=False,
    )

    kwargs = {
        "template": template,
        "max_batch_size": 1,
    }

    if adapter_path:
        kwargs["adapters"] = [adapter_path]

    engine = TransformersEngine(
        model_path,
        **kwargs,
    )

    return engine


def warmup(engine):
    request = InferRequest(
        messages=[
            {
                "role": "user",
                "content": "请回答：1+1等于多少？",
            }
        ]
    )

    config = RequestConfig(
        max_tokens=16,
        temperature=0,
    )

    engine.infer(
        [request],
        config,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_one(
    engine,
    tokenizer,
    question: str,
    max_new_tokens: int,
):
    request = InferRequest(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ]
    )

    config = RequestConfig(
        max_tokens=max_new_tokens,
        temperature=0,
        stream=True,
    )

    metric = InferStats()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    generators = engine.infer(
        [request],
        config,
        metrics=[metric],
    )

    first_token_time = None
    chunks = []

    for response in generators[0]:
        if response is None:
            continue

        content = response.choices[0].delta.content

        if not content:
            continue

        if first_token_time is None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            first_token_time = time.perf_counter()

        chunks.append(content)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    answer = "".join(chunks)

    output_tokens = len(
        tokenizer.encode(
            answer,
            add_special_tokens=False,
        )
    )

    total_time = end - start

    if first_token_time is None:
        ttft = total_time
        generation_time = total_time
    else:
        ttft = first_token_time - start
        generation_time = end - first_token_time

    tokens_per_second = (
        output_tokens / generation_time
        if generation_time > 0
        else 0
    )

    return {
        "question": question,
        "answer": answer,
        "output_tokens": output_tokens,
        "ttft_seconds": ttft,
        "total_seconds": total_time,
        "tokens_per_second": tokens_per_second,
        "swift_metrics": metric.compute(),
    }


def run_model(
    name: str,
    model_path: str,
    adapter_path: str | None,
    tokenizer,
    questions,
    max_new_tokens: int,
):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    engine = create_engine(
        model_path,
        adapter_path,
    )

    print("Warmup...")
    warmup(engine)

    results = []

    for index, question in enumerate(
        questions,
        start=1,
    ):
        print()
        print("-" * 80)
        print(f"Question {index}")
        print("-" * 80)
        print(question)

        result = infer_one(
            engine,
            tokenizer,
            question,
            max_new_tokens,
        )

        results.append(result)

        print()
        print("Answer:")
        print(result["answer"])

        print()
        print(
            f"Output tokens : "
            f"{result['output_tokens']}"
        )

        print(
            f"TTFT          : "
            f"{result['ttft_seconds']:.3f} s"
        )

        print(
            f"Total time    : "
            f"{result['total_seconds']:.3f} s"
        )

        print(
            f"Generation    : "
            f"{result['tokens_per_second']:.2f} tokens/s"
        )

        print(
            "Swift metrics :",
            result["swift_metrics"],
        )

    avg_ttft = sum(
        r["ttft_seconds"]
        for r in results
    ) / len(results)

    total_tokens = sum(
        r["output_tokens"]
        for r in results
    )

    total_generation_time = sum(
        max(
            r["total_seconds"]
            - r["ttft_seconds"],
            1e-9,
        )
        for r in results
    )

    summary = {
        "name": name,
        "average_ttft_seconds": avg_ttft,
        "overall_tokens_per_second":
            total_tokens / total_generation_time,
        "results": results,
    }

    del engine
    cleanup()

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--adapter",
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--output",
        default="benchmark_results.json",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print("Model:")
    print(args.model)

    print("Adapter:")
    print(args.adapter)

    print()
    print("CUDA:")
    print(torch.cuda.get_device_name(0))

    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(
                0
            ).total_memory / 1024**3,
            2,
        ),
        "GB",
    )

    # --------------------------------------------------------
    # 原始 Qwen3
    # --------------------------------------------------------

    base_result = run_model(
        name="BASE Qwen3-8B",
        model_path=args.model,
        adapter_path=None,
        tokenizer=tokenizer,
        questions=DEFAULT_QUESTIONS,
        max_new_tokens=args.max_new_tokens,
    )

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    lora_result = run_model(
        name="Qwen3-8B + TCM LoRA",
        model_path=args.model,
        adapter_path=args.adapter,
        tokenizer=tokenizer,
        questions=DEFAULT_QUESTIONS,
        max_new_tokens=args.max_new_tokens,
    )

    result = {
        "base": base_result,
        "lora": lora_result,
    }

    output_path = Path(args.output)

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        "Base average TTFT:",
        f"{base_result['average_ttft_seconds']:.3f}s",
    )

    print(
        "LoRA average TTFT:",
        f"{lora_result['average_ttft_seconds']:.3f}s",
    )

    print(
        "Base generation speed:",
        f"{base_result['overall_tokens_per_second']:.2f} tokens/s",
    )

    print(
        "LoRA generation speed:",
        f"{lora_result['overall_tokens_per_second']:.2f} tokens/s",
    )

    print()
    print("Saved:")
    print(output_path)


if __name__ == "__main__":
    main()

