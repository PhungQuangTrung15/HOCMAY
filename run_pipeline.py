from src.utils import set_seed, ensure_directories
from src.train_evaluate import run_experiment_pipeline


def main():
    print("=== BẮT ĐẦU CHẠY TOÀN BỘ PIPELINE DỰ ĐOÁN BỎ HỌC ===")
    set_seed(42)
    ensure_directories()
    results_df = run_experiment_pipeline()
    print("\n=== KẾT QUẢ SO SÁNH MÔ HÌNH ===")
    print(results_df.to_string(index=False))
    print("\n[OK] Hoàn tất! Kết quả và biểu đồ đã được lưu trong thư mục outputs/ và models/.")


if __name__ == "__main__":
    main()