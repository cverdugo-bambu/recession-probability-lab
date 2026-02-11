from src.recession_project.pipeline import run_pipeline


if __name__ == "__main__":
    metrics = run_pipeline()
    print("Pipeline complete. Test metrics:")
    print(f"selected_model: {metrics['selected_model']}")
    for key, value in metrics["selected_test_metrics"].items():
        print(f"{key}: {value}")
