"""
Urban Sound Classification 主入口
支持完整的训练-推理-提交流程
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import CONFIG, SAVE_DIR, IS_KAGGLE
from src.utils import seed_everything, print_config, cleanup_memory
from src.preprocessing import preprocess_all
from src.train import train_multi_resolution, merge_pseudo_labels
from src.inference import ensemble_inference, generate_submission
from src.evaluate import local_validation


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Urban Sound Classification")
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='all',
        choices=['preprocess', 'train', 'pseudo', 'inference', 'evaluate', 'all'],
        help='运行模式'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='CRNN',
        choices=['CRNN', 'EffNet'],
        help='模型类型'
    )
    parser.add_argument(
        '--resolutions',
        type=int,
        nargs='+',
        default=None,
        help='训练分辨率列表 (如: 64 96 128)'
    )
    parser.add_argument(
        '--no-tta',
        action='store_true',
        help='禁用 TTA'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='输出文件名'
    )
    
    return parser.parse_args()


def run_preprocessing():
    """运行数据预处理"""
    print("\n" + "=" * 50)
    print("🔧 阶段 1: 数据预处理")
    print("=" * 50)
    
    preprocess_all()
    print("✅ 预处理完成")


def run_training(model_type: str = "CRNN", resolutions: list = None):
    """运行模型训练"""
    print("\n" + "=" * 50)
    print(f"🎯 阶段 2: 训练 {model_type}")
    print("=" * 50)
    
    if resolutions is None:
        resolutions = CONFIG["FINAL_RESOLUTIONS"]
    
    train_multi_resolution(
        model_type=model_type,
        resolutions=resolutions,
        epochs=CONFIG["EPOCHS"],
        use_pseudo=False
    )
    
    print("✅ 训练完成")


def run_pseudo_labeling(model_type: str = "CRNN", resolutions: list = None):
    """运行伪标签训练"""
    print("\n" + "=" * 50)
    print(f"📝 阶段 3: 伪标签训练 {model_type}")
    print("=" * 50)
    
    if resolutions is None:
        resolutions = CONFIG["FINAL_RESOLUTIONS"]
    
    # 生成伪标签
    from src.train import create_pseudo_labels
    pseudo_df = create_pseudo_labels(model_type, resolutions)
    
    if pseudo_df is not None:
        # 合并数据
        merged_df = merge_pseudo_labels(pseudo_df)
        
        # 使用伪标签重新训练
        train_multi_resolution(
            model_type=model_type,
            resolutions=resolutions,
            epochs=CONFIG["EPOCHS"],
            use_pseudo=True,
            pseudo_df=merged_df
        )
    
    print("✅ 伪标签训练完成")


def run_inference(use_tta: bool = True, output_path: str = "submission.csv"):
    """运行推理"""
    print("\n" + "=" * 50)
    print("🔮 阶段 4: 集成推理")
    print("=" * 50)
    
    probs = ensemble_inference(use_tta=use_tta)
    
    submission = generate_submission(
        probs,
        output_path=Path(output_path),
        apply_postprocess=True
    )
    
    print(f"✅ 提交文件已生成: {output_path}")
    return submission


def run_evaluation(model_type: str = "CRNN", resolutions: list = None):
    """运行本地评估"""
    print("\n" + "=" * 50)
    print("📊 阶段 5: 本地评估")
    print("=" * 50)
    
    if resolutions is None:
        resolutions = CONFIG["FINAL_RESOLUTIONS"]
    
    results = local_validation(model_type, resolutions)
    
    print("✅ 评估完成")
    return results


def run_full_pipeline():
    """运行完整流程"""
    print("\n" + "🚀" * 20)
    print("    Urban Sound Classification - Full Pipeline")
    print("🚀" * 20)
    
    # 1. 预处理
    run_preprocessing()
    cleanup_memory()
    
    # 2. 训练 CRNN
    run_training("CRNN")
    cleanup_memory()
    
    # 3. 训练 EffNet (可选)
    if "EffNet" in CONFIG["MODELS"]:
        run_training("EffNet")
        cleanup_memory()
    
    # 4. 伪标签训练 (可选)
    if CONFIG.get("USE_PSEUDO", True):
        run_pseudo_labeling("CRNN")
        cleanup_memory()
    
    # 5. 本地评估
    run_evaluation()
    
    # 6. 推理生成提交
    run_inference()
    
    print("\n" + "🎉" * 20)
    print("    Pipeline 完成!")
    print("🎉" * 20)


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    seed = args.seed if args.seed else CONFIG["SEED"]
    seed_everything(seed)
    
    # 打印配置
    print_config()
    
    # 确保保存目录存在
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 根据模式运行
    if args.mode == 'preprocess':
        run_preprocessing()
    
    elif args.mode == 'train':
        run_training(args.model, args.resolutions)
    
    elif args.mode == 'pseudo':
        run_pseudo_labeling(args.model, args.resolutions)
    
    elif args.mode == 'inference':
        run_inference(
            use_tta=not args.no_tta,
            output_path=args.output
        )
    
    elif args.mode == 'evaluate':
        run_evaluation(args.model, args.resolutions)
    
    elif args.mode == 'all':
        run_full_pipeline()
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
